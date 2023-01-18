import argparse
import os
import shutil
import time
import errno
import math
import numpy as np
import wandb
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import networks.resnet
from imagenet32 import Imagenet32
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='InfoPro-PyTorch')

parser.add_argument('--eval-ensemble', dest='eval_ensemble', action='store_true',
                    help='If using ensemble (avg softmax) at evaluation time')
parser.set_defaults(eval_ensemble=False)
parser.add_argument('--ensemble-type', default='layerwise', type=str,
                    help='ensemble-type: [layerwise|last_layer_only]')
## Split arguments
parser.add_argument('--wide-list', default='16,16,32,64', type=str,
                    help='model width')
parser.add_argument('--groups', default=1, type=int,
                    help='Splitting each layer in this many sub-modules (1 is no split)')
parser.add_argument('--detach', dest='detach', action='store_true',
                    help='If using concatenation with detaching idea')
parser.add_argument('--detach-ratio', default=1.0, type=float,
                    help='Only used when gradient decoupling is used')
parser.set_defaults(detach=False)
parser.add_argument('--div-reg', dest='div_reg', action='store_true',
                    help='If using diversity regularizer (at softmax)')
parser.set_defaults(div_reg=False)
parser.add_argument('--div-temp', default=3.0, type=float,
                    help='diversity temperature (default: 3.0)')
parser.add_argument('--div-weight', default=0.5, type=float,
                    help='diversity_weight (default: 0.5)')

parser.add_argument('--infopro', dest='infopro', action='store_true',
                    help='If using infopro loss')
parser.set_defaults(infopro=False)

parser.add_argument('--lambdas', dest='lambdas', action='store_true',
                    help='If using lambda weights in loss function')
parser.set_defaults(lambdas=False)

parser.add_argument('--no-log', dest='no_log', action='store_true',
                    help='do not log if this is set true')
parser.set_defaults(no_log=False)


parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset: [cifar10|stl10|svhn]')

parser.add_argument('--model', default='resnet', type=str,
                    help='resnet is supported currently')

parser.add_argument('--layers', default=0, type=int,
                    help='total number of layers (have to be explicitly given!)')

parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.set_defaults(augment=True)

parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--name', default='', type=str,
                    help='name of experiment')
parser.add_argument('--no', default='1', type=str,
                    help='index of the experiment (for recording convenience)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')

# Cosine learning rate
parser.add_argument('--cos_lr', dest='cos_lr', action='store_true',
                    help='whether to use cosine learning rate')
parser.set_defaults(cos_lr=False)

parser.add_argument('--no_cuda', dest='no_cuda', action='store_true',
                    help='')
parser.set_defaults(no_cuda=False)

# InfoPro
parser.add_argument('--local_module_num', default='1', type=str,
                    help='number of local modules (1 refers to end-to-end training)')

parser.add_argument('--balanced_memory', dest='balanced_memory', action='store_true',
                    help='whether to split local modules with balanced GPU memory (InfoPro* in the paper)')
parser.set_defaults(balanced_memory=False)

parser.add_argument('--aux_net_config', default='1c2f', type=str,
                    help='architecture of auxiliary classifier / contrastive head '
                         '(default: 1c2f; 0c1f refers to greedy SL)'
                         '[0c1f|0c2f|1c1f|1c2f|1c3f|2c2f]')

parser.add_argument('--local_loss_mode', default='contrast', type=str,
                    help='ways to estimate the task-relevant info I(x, y)'
                         '[contrast|cross_entropy]')

parser.add_argument('--aux_net_widen', default=1.0, type=float,
                    help='widen factor of the two auxiliary nets (default: 1.0)')

parser.add_argument('--aux_net_feature_dim', default=128, type=int,
                    help='number of hidden features in auxiliary classifier / contrastive head '
                         '(default: 128)')

# The hyper-parameters \lambda_1 and \lambda_2 for 1st and (K-1)th local modules.
# Note that we assume they change linearly between these two modules.
# (The last module always uses standard end-to-end loss)
# See our paper for more details.

parser.add_argument('--ixx_1', default=1.0, type=float,)   # \lambda_1 for 1st local module
parser.add_argument('--ixy_1', default=1.0, type=float,)   # \lambda_2 for 1st local module

parser.add_argument('--ixx_2', default=1.0, type=float,)   # \lambda_1 for (K-1)th local module
parser.add_argument('--ixy_2', default=1.0, type=float,)   # \lambda_2 for (K-1)th local module

# data paths
parser.add_argument('--path', default='../masters_data', type=str, help='path')
parser.add_argument('--data-path ', default='../../data', type=str, help='data_path')

args = parser.parse_args()

# Configurations adopted for training deep networks.
training_configurations = {
    'resnet': {
        'epochs': 160,
        'batch_size': 1024, #if args.dataset in ['cifar10', 'svhn'] else 128,
        'initial_learning_rate': 0.8, #if args.dataset in ['cifar10', 'svhn'] else 0.1,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    }
}

wide_list = tuple(map(int, args.wide_list.split(",")))

exp_name = ('InfoPro*_' if args.balanced_memory else 'InfoPro_') \
              + str(args.dataset) \
              + '_' + str(args.model) + str(args.layers) \
              + '_K_' + str(args.local_module_num) \
              + '_G_' + str(args.groups) \
              + '_' + str(args.local_loss_mode) \
              + ('_wider_' + args.wide_list if wide_list[0] > 16 else '') \
              + ('_infopro' if args.infopro else '') \
              + ('_ixx_1_' + str(args.ixx_1) if args.infopro else '') \
              + ('_ixx_2_' + str(args.ixx_2) if args.infopro else '') \
              + ('_ixy_1_' + str(args.ixy_1) if args.infopro else '') \
              + ('_ixy_2_' + str(args.ixy_2) if args.infopro else '') \
              + (('_detach_' + str(args.detach_ratio)) if args.detach else '') \
              + ('_div' if args.div_reg else '') \
              + (('_T_' + str(args.div_temp)) if args.div_reg else '') \
              + (('_W_' + str(args.div_weight)) if args.div_reg else '') \
              + ('_new_config') \
              + ('_1c2f_stride=2') \
              + (('_ens(no head)_' + str(args.ensemble_type)) if args.eval_ensemble else '')
record_path = './logs/' + exp_name

print(exp_name)
# record_file = record_path + '/training_process.txt'
# accuracy_file = record_path + '/accuracy_epoch.txt'
# loss_file = record_path + '/loss_epoch.txt'
# check_point = os.path.join(record_path, args.checkpoint)

cuda = not args.no_cuda and torch.cuda.is_available()

def main():
    if not args.no_log:
		# add your wandb api key below
        wandb.login(key="")
        wandb.init(project='DGL-splits-resnet', entity='DGL', name=exp_name)
        config = wandb.config
        config.args = args

    global best_prec1
    best_prec1 = 0
    global val_acc
    val_acc = []

    class_num = args.dataset in ['cifar10', 'stl10', 'svhn'] and 10 or 100

    is_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda else "cpu")


    if 'cifar' in args.dataset:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        kwargs_dataset_train = {'train': True}
        kwargs_dataset_test = {'train': False}
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        image_size = 32
        num_classes = 10 if args.dataset == 'cifar10' else 100
    elif 'imagenet32' in args.dataset:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        kwargs_dataset_train = {'train': True}
        kwargs_dataset_test = {'train': False}
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        image_size = 32
        num_classes = 1000
    elif 'stl' in args.dataset:
        normalize = transforms.Normalize(mean=[x / 255 for x in [127.5, 127.5, 127.5]],
                                         std=[x / 255 for x in [127.5, 127.5, 127.5]])
        kwargs_dataset_train = {'split': 'train'}
        kwargs_dataset_test = {'split': 'test'}
        transform_train = transforms.Compose(
            [transforms.RandomCrop(96, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize])
        image_size = 96
        num_classes = 10
    elif 'svhn' in args.dataset:
        normalize = transforms.Normalize(mean=[x / 255 for x in [127.5, 127.5, 127.5]],
                                         std=[x / 255 for x in [127.5, 127.5, 127.5]])
        kwargs_dataset_train = {'split': 'train'}
        kwargs_dataset_test = {'split': 'test'}
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=2),
             transforms.ToTensor(),
             normalize])
        image_size = 32
        num_classes = 10
    elif 'FashionMNIST' in args.dataset:
        normalize =  transforms.Normalize((0.5,), (0.5,))
        kwargs_dataset_train = {'train': True}
        kwargs_dataset_test = {'train': False}
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              normalize])
        image_size = 28
        num_classes = 10
    else:
        raise NotImplementedError

    # Define test transforms
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 12, 'pin_memory': False}
    if args.dataset == "imagenet32":
        train_loader = torch.utils.data.DataLoader(Imagenet32(
            root=args.path+"/imagenet32/out_data_train", transform=transform_train, **kwargs_dataset_train),
            batch_size=training_configurations[args.model]['batch_size'], shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            Imagenet32(root=args.path+"/imagenet32/out_data_val", transform=transform_test,
                                                    **kwargs_dataset_test),
            batch_size = training_configurations[args.model]['batch_size'], shuffle = False, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()](root=args.path, download=True, transform=transform_train,
                                                    **kwargs_dataset_train),
            batch_size=training_configurations[args.model]['batch_size'], shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()](root=args.path, transform=transform_test,
                                                    **kwargs_dataset_test),
            batch_size=training_configurations[args.model]['batch_size'], shuffle=False, **kwargs)

    # create model
    if args.model == 'resnet':
        model = eval('networks.resnet.resnet' + str(args.layers))\
            (local_module_num=args.local_module_num,
             batch_size=training_configurations[args.model]['batch_size'],
             image_size=image_size,
             balanced_memory=args.balanced_memory,
             dataset=args.dataset,
             class_num=num_classes,
             wide_list=wide_list,
             dropout_rate=args.droprate,
             aux_net_config=args.aux_net_config,
             local_loss_mode=args.local_loss_mode,
             aux_net_widen=args.aux_net_widen,
             aux_net_feature_dim=args.aux_net_feature_dim,
             cuda=cuda, infopro = args.infopro,
             groups=args.groups, lambdas=args.lambdas,
             detach=args.detach, detach_ratio=args.detach_ratio, div_reg=args.div_reg,
             div_temp=args.div_temp, div_weight=args.div_weight, device=device)
    else:
        raise NotImplementedError

    print(model)

    # Flop Counter Code
    if args.model == 'resnet':
        flop_count_model = eval('networks.resnet.resnet' + str(args.layers)) \
            (local_module_num=args.local_module_num,
             batch_size=training_configurations[args.model]['batch_size'],
             image_size=image_size,
             balanced_memory=args.balanced_memory,
             dataset=args.dataset,
             class_num=num_classes,
             wide_list=wide_list,
             dropout_rate=args.droprate,
             aux_net_config=args.aux_net_config,
             local_loss_mode=args.local_loss_mode,
             aux_net_widen=args.aux_net_widen,
             aux_net_feature_dim=args.aux_net_feature_dim,
             cuda=cuda, infopro=args.infopro,
             groups=args.groups, lambdas=args.lambdas,
             detach=args.detach, detach_ratio=args.detach_ratio, div_reg=args.div_reg,
             div_temp=args.div_temp, div_weight=args.div_weight, device=torch.device("cpu"))
        flop_count_model.train()
        flop_count_model(img=torch.rand((1,3,image_size,image_size)),
              target=torch.tensor([0]), count_flops=True, eval_ensemble=args.eval_ensemble)

        flop_count_model.eval()
        flop_count_model(img=torch.rand((1, 3, image_size, image_size)),
          target=torch.tensor([0]), count_flops=True, eval_ensemble=args.eval_ensemble)

    ########

    # if not os.path.isdir(check_point):
    #     mkdir_p(check_point)

    cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(),
                                    lr=training_configurations[args.model]['initial_learning_rate'],
                                    momentum=training_configurations[args.model]['momentum'],
                                    nesterov=training_configurations[args.model]['nesterov'],
                                    weight_decay=training_configurations[args.model]['weight_decay']
                                )
    if cuda:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        val_acc = checkpoint['val_acc']
        best_prec1 = checkpoint['best_acc']
        np.savetxt(accuracy_file, np.array(val_acc))
    else:
        start_epoch = 0

    if not args.no_log:
        wandb.watch(model)

    for epoch in range(start_epoch, training_configurations[args.model]['epochs']):

        adjust_learning_rate(optimizer, epoch + 1)

        # train for one epoch
        train_loss, train_prec1, train_thread_prec1 = train(train_loader, model, optimizer, epoch)

        if not args.no_log:
            wandb.log({"Train Thread Loss": train_loss}, step=epoch)
            wandb.log({"Prec@1": train_prec1}, step=epoch)
            wandb.log({"Thread Prec@1": train_thread_prec1}, step=epoch)

        # evaluate on validation set
        val_loss, val_prec1, val_thread_prec1= validate(val_loader, model, epoch, args.eval_ensemble, args.ensemble_type)

        if not args.no_log:
            wandb.log({"Val Thread Loss": val_loss}, step=epoch)
            wandb.log({"Val Prec@1": val_prec1}, step=epoch)
            wandb.log({"Val Thread Prec@1": val_thread_prec1}, step=epoch)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_acc': best_prec1,
        #     'optimizer': optimizer.state_dict(),
        #     'val_acc': val_acc,
        #
        # }, is_best, checkpoint=check_point)
        # print('Best accuracy: ', best_prec1)
        # np.savetxt(accuracy_file, np.array(val_acc))

def train(train_loader, model, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    thread_top1 = AverageMeter()

    train_batches_num = len(train_loader)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):

        if cuda:
            target = target.cuda()
            x = x.cuda()

        optimizer.zero_grad()
        thread_output, output, loss = model(img=x,
                             target=target,
                             ixx_1=args.ixx_1,
                             ixy_1=args.ixy_1,
                             ixx_2=args.ixx_2,
                             ixy_2=args.ixy_2,
                             ensemble_type=args.ensemble_type,
                             eval_ensemble=args.eval_ensemble)

        optimizer.step()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        thread_prec1 = accuracy(thread_output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))
        thread_top1.update(thread_prec1.item(), x.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or (i+1) == train_batches_num:
            # print(discriminate_weights)
        #     fd = open(record_file, 'a+')
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Thread Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'
                      'Thread Prec@1 {thread_top1.value:.3f} ({thread_top1.ave:.3f})\t'.format(
                       epoch, i+1, train_batches_num, batch_time=batch_time,
                       loss=losses, top1=top1, thread_top1=thread_top1))
        #
            print(string)
        #     # print(weights)
        #     fd.write(string + '\n')
        #     fd.close()

    return losses.ave, top1.ave, thread_top1.ave

def validate(val_loader, model, epoch, eval_ensemble, ensemble_type):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    thread_top1 = AverageMeter()

    train_batches_num = len(val_loader)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        if cuda:
            target = target.cuda()
            input = input.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            thread_output, output, loss = model(img=input_var,
                                 target=target_var,
                                 eval_ensemble=eval_ensemble,
                                 ensemble_type=ensemble_type)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        thread_prec1 = accuracy(thread_output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        thread_top1.update(thread_prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # fd = open(record_file, 'a+')
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Thread Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'
              'Thread Prec@1 {thread_top1.value:.3f} ({thread_top1.ave:.3f})\t'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses, top1=top1, thread_top1=thread_top1))

    print(string)
    # fd.write(string + '\n')
    # fd.close()
    val_acc.append(top1.ave)

    return losses.ave, top1.ave, thread_top1.ave

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

    else:
        for param_group in optimizer.param_groups:
            if epoch <= 10:
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs'])) * (epoch - 1) / 10 + 0.01 * (11 - epoch) / 10
            else:
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                    * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs']))
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()

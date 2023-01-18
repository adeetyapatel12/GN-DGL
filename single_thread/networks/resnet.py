'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math
from utils import div_loss_calc, get_ensemble_logits
from ptflops import get_model_complexity_info
from networks.flop_count_dummy_models import Model_1, Model_2, print_flops

torch.manual_seed(0)

from .configs import InfoPro, InfoPro_balanced_memory, DWM_DGL
from .split_auxiliary_nets import Decoder, AuxClassifier

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, groups=groups, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0, first_conv_groups=1, groups=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, groups=first_conv_groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class InfoProResNet(nn.Module):

    def __init__(self, block, layers, arch, local_module_num, batch_size, image_size=32,
                 balanced_memory=False, dataset='cifar10', class_num=10,
                 wide_list=(16, 16, 32, 64), dropout_rate=0,
                 aux_net_config='1c2f', local_loss_mode='contrast',
                 aux_net_widen=1, aux_net_feature_dim=128, cuda=False,
                 infopro = False, groups=1, lambdas=False, detach=False, detach_ratio=1.0,
                 div_reg=False, div_temp=0.0, div_weight=1.0, device='cpu'):
        super(InfoProResNet, self).__init__()

        assert arch in ['resnet32', 'resnet110'], "This repo supports resnet32 and resnet110 currently. " \
                                                  "For other networks, please set network configs in .configs."

        try:
            if infopro:
                self.infopro_config = InfoPro_balanced_memory[arch][dataset][local_module_num] \
                    if balanced_memory else InfoPro[arch][local_module_num]
            else:
                self.infopro_config = DWM_DGL[arch][local_module_num]
                local_module_num = len(self.infopro_config) + 1
        except:
            raise NotImplementedError

        self.infopro = infopro
        self.lambdas = lambdas
        self.detach = detach
        self.detach_ratio = detach_ratio
        self.div_reg = div_reg
        self.div_temp = div_temp
        self.div_weight = div_weight
        self.device = device

        self.inplanes = wide_list[0]
        self.dropout_rate = dropout_rate
        self.feature_num = wide_list[-1]
        self.class_num = class_num
        self.local_module_num = int(local_module_num)
        self.layers = layers
        self.groups = groups
        self.wide_list = wide_list
        self.aux_net_widen  = wide_list[0] / 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, wide_list[1], layers[0], groups=self.groups, stage=1)
        self.layer2 = self._make_layer(block, wide_list[2], layers[1], groups=self.groups, stride=2, stage=2)
        self.layer3 = self._make_layer(block, wide_list[3], layers[2], groups=self.groups, stride=2, stage=3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_num, self.class_num)

        ## Adding head at last same as auxilliary network
        inplanes = self.feature_num
        feature_dim = aux_net_feature_dim
        self.head = nn.Sequential(
            nn.Conv2d(inplanes, int(inplanes * 2), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(inplanes * 2)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(int(inplanes * 2), int(feature_dim * self.aux_net_widen)),
            nn.ReLU(inplace=True),
            nn.Linear(int(feature_dim * self.aux_net_widen), self.class_num)
        )
        #############

        self.criterion_ce = nn.CrossEntropyLoss()

        for item in self.infopro_config:
            module_index, layer_index = item
            for group_index in range(self.groups):
                if self.detach:
                    if self.groups==1 or self.detach_ratio == 1.0:
                        if self.infopro:
                            exec('self.decoder_' + str(module_index) + '_' + str(layer_index) + '_' + str(group_index) +
                                 '= Decoder(wide_list[module_index], image_size, widen=self.aux_net_widen)')

                        exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) + '_' + str(group_index) +
                             '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                             'loss_mode=local_loss_mode, class_num=class_num, '
                             'widen=self.aux_net_widen, feature_dim=aux_net_feature_dim)')
                    else:
                        self.aux_in_channels_per_group = int(wide_list[module_index]/self.groups) + int(self.detach_ratio * (self.groups-1) * (wide_list[module_index]/self.groups))
                        self.aux_net_feature_dim_per_group = int(aux_net_feature_dim/self.groups) + int(self.detach_ratio * (self.groups-1) * (aux_net_feature_dim/self.groups))
                        if self.infopro:
                            exec('self.decoder_' + str(module_index) + '_' + str(layer_index) + '_' + str(group_index) +
                                 '= Decoder(self.aux_in_channels_per_group, image_size, widen=self.aux_net_widen)')

                        exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) + '_' + str(
                            group_index) +
                             '= AuxClassifier(self.aux_in_channels_per_group, net_config=aux_net_config, '
                             'loss_mode=local_loss_mode, class_num=class_num, '
                             'widen=self.aux_net_widen, feature_dim=self.aux_net_feature_dim_per_group)')
                else:
                    if self.infopro:
                        exec('self.decoder_' + str(module_index) + '_' + str(layer_index) + '_' + str(group_index) +
                             '= Decoder(int(wide_list[module_index]/self.groups), image_size, widen=self.aux_net_widen)')

                    exec('self.aux_classifier_' + str(module_index) + '_' + str(layer_index) + '_' + str(group_index) +
                         '= AuxClassifier(int(wide_list[module_index]/self.groups), net_config=aux_net_config, '
                         'loss_mode=local_loss_mode, class_num=class_num, '
                         'widen=self.aux_net_widen, feature_dim=int(aux_net_feature_dim/self.groups))')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if 'cifar' in dataset:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            )
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            )
        else:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            )
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            )

        if cuda:
            self.mask_train_mean = self.mask_train_mean.cuda()
            self.mask_train_std = self.mask_train_std.cuda()

    def _image_restore(self, normalized_image):
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
               + self.mask_train_mean[:normalized_image.size(0)]

    def _get_local_mod_boundaries(self, stage):
        boundaries = []
        for item in self.infopro_config:
            module_index, layer_index = item
            if module_index == stage:
                boundaries.append(layer_index)
        return boundaries

    def _make_layer(self, block, planes, blocks, groups=1, stride=1, stage=None):

        layers = []
        first_conv_groups = groups
        if (0 in self._get_local_mod_boundaries(stage-1) if stage == 1 else (self.layers[stage-2]-1) in self._get_local_mod_boundaries(stage-1)):
            first_conv_groups = 1

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, groups=first_conv_groups, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate, first_conv_groups=first_conv_groups, groups=groups))

        self.inplanes = planes * block.expansion
        for b in range(1, blocks):
            first_conv_groups = groups
            if (b-1) in self._get_local_mod_boundaries(stage):
                first_conv_groups = 1
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate, first_conv_groups=first_conv_groups, groups=groups))

        return nn.Sequential(*layers)

    def initialize_flop_counter(self):
      flop_counter = {}
      for k in range(self.local_module_num - 1):
          for g in range(self.groups):
            flop_counter[str(k) + '_' + str(g)] = 0
      flop_counter['last_head'] = 0
      return flop_counter

    def forward_original(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, img, target=None,
                ixx_1=0, ixy_1=0,
                ixx_2=0, ixy_2=0,
                count_flops=False,
                eval_ensemble=False,
                ensemble_type='layerwise'):
        str_x = ""
        all_logits = {}
        if count_flops:
            flop_counter = self.initialize_flop_counter()

        if self.training or eval_ensemble:
            stage_i = 0
            layer_i = 0
            local_module_i = 0


            ## Flop counter code
            if count_flops:
                macs, _= get_model_complexity_info(nn.Sequential(self.conv1, self.bn1, self.relu),
                                                     (img.shape[1], img.shape[2], img.shape[3]),
                                                     as_strings=False, print_per_layer_stat=False, verbose=False)

                for group_i in range(self.groups):
                    flop_counter[str(local_module_i) + '_' + str(group_i)] += macs/ self.groups
            #######

            x = self.conv1(img)
            x = self.bn1(x)
            x = self.relu(x)

            if local_module_i <= self.local_module_num - 2:
                if self.infopro_config[local_module_i][0] == stage_i \
                        and self.infopro_config[local_module_i][1] == layer_i:
                    ratio = local_module_i / (self.local_module_num - 2) if self.local_module_num > 2 else 0

                    loss_ixx = 0
                    loss_ixy = 0
                    if self.infopro:
                        assert self.groups==1, "Infopro is not supported with groups more than 1."
                        ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio if self.lambdas else 1
                        ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio if self.lambdas else 1
                        for group_i in range(self.groups):

                            # Flop counter code
                            if count_flops:
                                macs, _ = get_model_complexity_info(eval('self.decoder_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i)),
                                                                    (x.shape[1], x.shape[2], x.shape[3]),
                                                                    as_strings=False, print_per_layer_stat=False,
                                                                    verbose=False)
                                flop_counter[str(local_module_i) + '_' + str(group_i)] += macs

                                macs, _ = get_model_complexity_info(eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i)),
                                                                    (x.shape[1], x.shape[2], x.shape[3]),
                                                                    as_strings=False, print_per_layer_stat=False,
                                                                    verbose=False)
                                flop_counter[str(local_module_i) + '_' + str(group_i)] += macs

                            #######

                            loss_ixx += eval('self.decoder_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i))(x, self._image_restore(img), for_flop_count=False)
                            loss_temp, _ = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i))(x, target)
                            loss_ixy += loss_temp
                        loss = ixx_r * loss_ixx + ixy_r * loss_ixy
                    else:
                        ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio if self.lambdas else 1
                        aux_in_channels = int(self.wide_list[stage_i]/self.groups)
                        total_channels = self.wide_list[stage_i]

                        group_logits = []
                        for group_i in range(self.groups):
                            ch_from = aux_in_channels * group_i
                            ch_to = aux_in_channels * (group_i + 1)
                            if self.detach:
                                if self.groups==1 or self.detach_ratio == 1.0:
                                    x_detached = x.clone().detach()
                                    x_detached[:, ch_from:ch_to, :, :] = x[:, ch_from:ch_to, :, :].clone()

                                    # Flop counter code
                                    if count_flops:
                                        macs, _ = get_model_complexity_info(eval(
                                            'self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i)),
                                            (x_detached.shape[1], x_detached.shape[2], x_detached.shape[3]),
                                            as_strings=False, print_per_layer_stat=False,
                                            verbose=False)
                                        flop_counter[str(local_module_i) + '_' + str(group_i)] += macs
                                    #########

                                    loss_temp, logits = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i))(x_detached, target)
                                    loss_ixy += loss_temp
                                    group_logits.append(logits)
                                else:
                                    x_detached = []
                                    current_ch_from = 0
                                    for i in range(self.groups):
                                        if i == group_i:
                                            x_detached.append(x[:, ch_from:ch_to, :, :].clone())
                                        else:
                                            ch_from = aux_in_channels * i
                                            ch_to = aux_in_channels * (i + 1)
                                            x_detached_sliced = x[:, ch_from:ch_to, :, :].clone().detach()
                                            current_ch_from = current_ch_from % (ch_to-ch_from)
                                            no_channels = int(self.detach_ratio * (ch_to-ch_from))
                                            x_detached.append(x_detached_sliced[:,current_ch_from:current_ch_from+no_channels ,:,:])
                                            current_ch_from = current_ch_from + no_channels
                                    x_detached = torch.cat(x_detached, dim=1)
                                    # Flop counter code
                                    if count_flops:
                                        macs, _ = get_model_complexity_info(eval(
                                            'self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(
                                                group_i)),
                                            (x_detached.shape[1], x_detached.shape[2], x_detached.shape[3]),
                                            as_strings=False, print_per_layer_stat=False,
                                            verbose=False)
                                        flop_counter[str(local_module_i) + '_' + str(group_i)] += macs
                                    #########

                                    loss_temp, logits = eval(
                                        'self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(
                                            group_i))(x_detached, target)
                                    loss_ixy += loss_temp
                                    group_logits.append(logits)
                            else:
                                # Flop counter code
                                if count_flops:
                                    macs, _ = get_model_complexity_info(eval(
                                        'self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i)),
                                        (x[:, ch_from:ch_to, :, :].shape[1], x[:, ch_from:ch_to, :, :].shape[2], x[:, ch_from:ch_to, :, :].shape[3]),
                                        as_strings=False, print_per_layer_stat=False,
                                        verbose=False)
                                    flop_counter[str(local_module_i) + '_' + str(group_i)] += macs
                                #########

                                loss_temp, logits = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i))(x[:, ch_from:ch_to, :, :], target)
                                loss_ixy += loss_temp
                                group_logits.append(logits)

                        all_logits[str(local_module_i)] = group_logits
                        loss = ixy_r * loss_ixy
                        if self.div_reg:
                            div_loss = div_loss_calc(group_logits, target, self.div_temp, self.device)
                            loss += self.div_weight * div_loss

                    if self.training:
                        loss.backward()
                    x = x.detach()
                    local_module_i += 1

                    # # Test of splits
                    # print()
                    # print(stage_i, layer_i)
                    # for a in range(list(self.conv1.parameters())[0].shape[0]):
                    #     print(a, torch.sum(abs(list(self.conv1.parameters())[0].grad[a])))

            for stage_i in (1, 2, 3):
                for layer_i in range(self.layers[stage_i - 1]):

                    # Flop counter code
                    if count_flops:
                            macs, _ = get_model_complexity_info(eval('self.layer' + str(stage_i))[layer_i],
                                        (x.shape[1], x.shape[2], x.shape[3]),
                                        as_strings=False, print_per_layer_stat=False,
                                        verbose=False)
                            for group_i in range(self.groups):
                                flop_counter[str(local_module_i) + '_' + str(group_i)] += macs / self.groups
                    #########
                    x = eval('self.layer' + str(stage_i))[layer_i](x)

                    if local_module_i <= self.local_module_num - 2:
                        if self.infopro_config[local_module_i][0] == stage_i \
                                and self.infopro_config[local_module_i][1] == layer_i:
                            ratio = local_module_i / (self.local_module_num - 2) if self.local_module_num > 2 else 0

                            loss_ixx = 0
                            loss_ixy = 0
                            if self.infopro:
                                assert self.groups==1, "Infopro is not supported with groups more than 1."
                                ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio if self.lambdas else 1
                                ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio if self.lambdas else 1
                                for group_i in range(self.groups):

                                    # Flop counter code
                                    if count_flops:
                                        macs, _ = get_model_complexity_info(eval('self.decoder_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i)),
                                                                            (x.shape[1], x.shape[2], x.shape[3]),
                                                                            as_strings=False, print_per_layer_stat=False,
                                                                            verbose=False)
                                        flop_counter[str(local_module_i) + '_' + str(group_i)] += macs

                                        macs, _ = get_model_complexity_info(eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i)),
                                                                            (x.shape[1], x.shape[2], x.shape[3]),
                                                                            as_strings=False, print_per_layer_stat=False,
                                                                            verbose=False)
                                        flop_counter[str(local_module_i) + '_' + str(group_i)] += macs

                                    #######

                                    loss_ixx += eval('self.decoder_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i))(x, self._image_restore(img), for_flop_count=False)
                                    loss_temp, _ = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i))(x, target)
                                    loss_ixy += loss_temp
                                loss = ixx_r * loss_ixx + ixy_r * loss_ixy
                            else:

                                ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio if self.lambdas else 1
                                aux_in_channels = int(self.wide_list[stage_i] / self.groups)

                                group_logits = []
                                for group_i in range(self.groups):
                                    ch_from = aux_in_channels * group_i
                                    ch_to = aux_in_channels * (group_i + 1)
                                    if self.detach:
                                        if self.groups == 1 or self.detach_ratio == 1.0:
                                            x_detached = x.clone().detach()
                                            x_detached[:, ch_from:ch_to, :, :] = x[:, ch_from:ch_to, :, :].clone()

                                            # Flop counter code
                                            if count_flops:
                                                macs, _ = get_model_complexity_info(eval(
                                                    'self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i)),
                                                    (x_detached.shape[1], x_detached.shape[2], x_detached.shape[3]),
                                                    as_strings=False, print_per_layer_stat=False,
                                                    verbose=False)
                                                flop_counter[str(local_module_i) + '_' + str(group_i)] += macs
                                            ##########

                                            loss_temp, logits = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i))(x_detached, target)
                                            loss_ixy += loss_temp
                                            group_logits.append(logits)
                                        else:
                                            x_detached = []
                                            current_ch_from = 0
                                            for i in range(self.groups):
                                                if i == group_i:
                                                    x_detached.append(x[:, ch_from:ch_to, :, :].clone())
                                                else:
                                                    ch_from = aux_in_channels * i
                                                    ch_to = aux_in_channels * (i + 1)
                                                    x_detached_sliced = x[:, ch_from:ch_to, :, :].clone().detach()
                                                    current_ch_from = current_ch_from % (ch_to - ch_from)
                                                    no_channels = int(self.detach_ratio * (ch_to - ch_from))
                                                    x_detached.append(x_detached_sliced[:,
                                                                      current_ch_from:current_ch_from + no_channels, :,
                                                                      :])
                                                    current_ch_from = current_ch_from + no_channels
                                            x_detached = torch.cat(x_detached, dim=1)
                                            # Flop counter code
                                            if count_flops:
                                                macs, _ = get_model_complexity_info(eval(
                                                    'self.aux_classifier_' + str(stage_i) + '_' + str(
                                                        layer_i) + '_' + str(
                                                        group_i)),
                                                    (x_detached.shape[1], x_detached.shape[2], x_detached.shape[3]),
                                                    as_strings=False, print_per_layer_stat=False,
                                                    verbose=False)
                                                flop_counter[str(local_module_i) + '_' + str(group_i)] += macs
                                            #########

                                            loss_temp, logits = eval(
                                                'self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(
                                                    group_i))(x_detached, target)
                                            loss_ixy += loss_temp
                                            group_logits.append(logits)
                                    else:
                                        # Flop counter code
                                        if count_flops:
                                            macs, _ = get_model_complexity_info(eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i)),
                                                (x[:, ch_from:ch_to, :, :].shape[1], x[:, ch_from:ch_to, :, :].shape[2], x[:, ch_from:ch_to, :, :].shape[3]),
                                                as_strings=False, print_per_layer_stat=False, verbose=False)
                                            flop_counter[str(local_module_i) + '_' + str(group_i)] += macs
                                        #########

                                        loss_temp, logits = eval('self.aux_classifier_' + str(stage_i) + '_' + str(layer_i) + '_' + str(group_i))(x[:, ch_from:ch_to, :, :], target)
                                        loss_ixy += loss_temp
                                        group_logits.append(logits)

                                all_logits[str(local_module_i)] = group_logits
                                loss = ixy_r * loss_ixy
                                if self.div_reg:
                                    div_loss = div_loss_calc(group_logits, target, self.div_temp, self.device)
                                    loss += self.div_weight * div_loss
                            if self.training:
                                loss.backward()
                            x = x.detach()
                            local_module_i += 1

                            # # Test of splits
                            # if layer_i >= 0:
                            #     print()
                            #     print(stage_i, layer_i)
                            #     for a in range(list(eval('self.layer' + str(stage_i))[layer_i].parameters())[0].shape[0]):
                            #         print(a, torch.sum(abs(list(eval('self.layer' + str(stage_i))[layer_i].parameters())[0].grad[a])))


            # Flop counter code
            # if count_flops:
            #     macs, _ = get_model_complexity_info(Model_1(self.avgpool, self.fc, self.criterion_ce),
            #                                         (x.shape[1], x.shape[2], x.shape[3]),
            #                                         as_strings=False, print_per_layer_stat=False,
            #                                         verbose=False)
            #     flop_counter["last_head"] += macs
            #########
            #
            # x = self.avgpool(x)
            # x = x.view(x.size(0), -1)
            # logits = self.fc(x)

            #### Flop counter code

            # if count_flops:
            #     macs, _ = get_model_complexity_info(self.head,
            #                                         (x.shape[1], x.shape[2], x.shape[3]),
            #                                         as_strings=False, print_per_layer_stat=False,
            #                                         verbose=False)
            #     flop_counter['last_head'] += macs

            ########

            # logits = self.head(x)
            # all_logits[str(local_module_i)] = [logits]
            # loss = self.criterion_ce(logits, target)
            # if self.training:
            #     loss.backward()

            # Flop counter code
            if count_flops:
                print("Training time stats: ")
                print_flops(flop_counter)
            #########

            if eval_ensemble:
                thread_logits, logits = get_ensemble_logits(all_logits, self.device, ensemble_type)
            return thread_logits, logits, loss

        else:

            if count_flops:
                m2 = Model_2(self.conv1, self.bn1, self.relu, self.layer1, self.layer2, self.layer3, self.head, self.criterion_ce)

                inference_flops, _ = get_model_complexity_info(m2, (img.shape[1], img.shape[2], img.shape[3]),
                                                as_strings=False, print_per_layer_stat=False,
                                                verbose=False)
                print("Inference time flops: ", inference_flops)

            x = self.conv1(img)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            # x = self.avgpool(x)
            # x = x.view(x.size(0), -1)
            # logits = self.fc(x)

            # logits = self.head(x)
            # loss = self.criterion_ce(logits, target)

            return logits, loss

def resnet20(**kwargs):
    model = InfoProResNet(BasicBlock, [3, 3, 3], arch='resnet20', **kwargs)
    return model

def resnet32(**kwargs):
    model = InfoProResNet(BasicBlock, [5, 5, 5], arch='resnet32', **kwargs)
    return model


def resnet44(**kwargs):
    model = InfoProResNet(BasicBlock, [7, 7, 7], arch='resnet44', **kwargs)
    return model


def resnet56(**kwargs):
    model = InfoProResNet(BasicBlock, [9, 9, 9], arch='resnet56', **kwargs)
    return model


def resnet110(**kwargs):
    model = InfoProResNet(BasicBlock, [18, 18, 18], arch='resnet110', **kwargs)
    return model


def resnet1202(**kwargs):
    model = InfoProResNet(BasicBlock, [200, 200, 200], arch='resnet1202', **kwargs)
    return model


def resnet164(**kwargs):
    model = InfoProResNet(Bottleneck, [18, 18, 18], arch='resnet164', **kwargs)
    return model


def resnet1001(**kwargs):
    model = InfoProResNet(Bottleneck, [111, 111, 111], arch='resnet1001', **kwargs)
    return model



# if __name__ == '__main__':
#     net = resnet32(local_module_num=16, batch_size=256, image_size=32,
#                  balanced_memory=False, dataset='cifar10', class_num=10,
#                  wide_list=(16, 16, 32, 64), dropout_rate=0,
#                  aux_net_config='1c2f', local_loss_mode='contrast',
#                  aux_net_widen=1, aux_net_feature_dim=128)
#     y = net(torch.randn(4, 3, 32, 32), torch.zeros(4).long())

    # print(net)
    # print(y.size())
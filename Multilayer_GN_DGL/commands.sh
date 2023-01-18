# resnet32 with cifar10 (different K and G combinations specified by "local_mod" and "group" variables)

datasets=('cifar10' 'cifar100' 'imagenet32')

local_module_num=('1A' '2' '4')
groups=(1 2 4 8)

#local_module_num=('8')
#groups=(1 2 4)

#local_module_num=('16')
#groups=(1 2)

for local_mod in "${local_module_num[@]}"
do
for group in "${groups[@]}"
do
for dataset in "${datasets[@]}"
do
python train.py --dataset $dataset --model resnet --layers 32 --droprate 0.0 --cos_lr  --local_module_num $local_mod --groups $group --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 16,16,32,64 --aux_net_feature_dim 128 --aux_net_config 1c2f --detach --detach-ratio 1.0 --div-reg --div-temp 3.0 --div-weight 0.5 --eval-ensemble --ensemble-type layerwise
done
done
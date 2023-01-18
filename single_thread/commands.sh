# resnet32 with cifar10, single thread for G=2
python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 2 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 16,16,32,64 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# resnet32 with cifar10, single thread for G=4
python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 4 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 16,16,32,64 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# resnet32 with cifar10, single thread for G=8
python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 8 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 16,16,32,64 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"



# wider resnet32x2 with cifar10, single thread for G=2
python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 2 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 32,32,64,128 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# wider resnet32x2 with cifar10, single thread for G=4
python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 4 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 32,32,64,128 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# wider resnet32x2 with cifar10, single thread for G=8
python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 8 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 32,32,64,128 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"





# resnet32 with cifar100, single thread for G=2
python train.py --dataset cifar100 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 2 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 16,16,32,64 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# resnet32 with cifar100, single thread for G=4
python train.py --dataset cifar100 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 4 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 16,16,32,64 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# resnet32 with cifar100, single thread for G=8
python train.py --dataset cifar100 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 8 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 16,16,32,64 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# wider resnet32x2 with cifar100, single thread for G=2
python train.py --dataset cifar100 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 2 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 32,32,64,128 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# wider resnet32x2 with cifar100, single thread for G=4
python train.py --dataset cifar100 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 4 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 32,32,64,128 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# wider resnet32x2 with cifar100, single thread for G=8
python train.py --dataset cifar100 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 8 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 32,32,64,128 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"



# resnet32 with imagenet32, single thread for G=2
python train.py --dataset imagenet32 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 2 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 16,16,32,64 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# resnet32 with imagenet32, single thread for G=4
python train.py --dataset imagenet32 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 4 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 16,16,32,64 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# resnet32 with imagenet32, single thread for G=8
python train.py --dataset imagenet32 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 8 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 16,16,32,64 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# wider resnet32x2 with imagenet32, single thread for G=2
python train.py --dataset imagenet32 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 2 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 32,32,64,128 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# wider resnet32x2 with imagenet32, single thread for G=4
python train.py --dataset imagenet32 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 4 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 32,32,64,128 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"

# wider resnet32x2 with imagenet32, single thread for G=8
python train.py --dataset imagenet32 --model resnet --layers 32 --droprate 0.0 --cos_lr --local_module_num 1A  --groups 8 --local_loss_mode cross_entropy --aux_net_widen 1 --wide-list 32,32,64,128 --aux_net_feature_dim 128 --aux_net_config 1c2f --eval-ensemble --ensemble-type layerwise"
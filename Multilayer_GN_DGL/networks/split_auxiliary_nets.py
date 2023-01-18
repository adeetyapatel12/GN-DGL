import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import SupConLoss

torch.manual_seed(0)

class Decoder(nn.Module):
    def __init__(self, inplanes, image_size, interpolate_mode='bilinear', widen=1, splits=1):
        super(Decoder, self).__init__()

        self.image_size = image_size
        self.splits = splits

        assert interpolate_mode in ['bilinear', 'nearest']
        self.interpolate_mode = interpolate_mode

        self.bce_loss = nn.BCELoss()

        self.decoder = nn.Sequential(
            nn.Conv2d(inplanes, int((12 * widen)/self.splits), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int((12 * widen)/self.splits)),
            nn.ReLU(),
            nn.Conv2d(int((12 * widen)/self.splits), 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features, image_ori=torch.rand((1, 3, 32, 32)), for_flop_count=True):

        if for_flop_count:
            features = torch.rand(features.shape)

        if self.interpolate_mode == 'bilinear':
            features = F.interpolate(features, size=[self.image_size, self.image_size],
                                     mode='bilinear', align_corners=True)
        elif self.interpolate_mode == 'nearest':   # might be faster
            features = F.interpolate(features, size=[self.image_size, self.image_size],
                                     mode='nearest')
        else:
            raise NotImplementedError


        return self.bce_loss(self.decoder(features), image_ori)

class AuxClassifier(nn.Module):
    def __init__(self, inplanes, net_config='1c2f', loss_mode='contrast', class_num=10, widen=1, feature_dim=128):
        super(AuxClassifier, self).__init__()

        # assert inplanes in [16, 32, 64]
        assert net_config in ['0c1f', '0c2f', '1c1f', '1c2f', '1c3f', '2c2f']
        assert loss_mode in ['contrast', 'cross_entropy']

        self.loss_mode = loss_mode
        self.feature_dim = feature_dim

        if loss_mode == 'contrast':
            self.criterion = SupConLoss()
            self.fc_out_channels = feature_dim
        elif loss_mode == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
            self.fc_out_channels = class_num
        else:
            raise NotImplementedError

        if net_config == '1c2f':
            self.head = nn.Sequential(
                nn.Conv2d(inplanes, int(inplanes * 2), kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(int(inplanes * 2)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(int(inplanes * 2), int(feature_dim * widen)),
                nn.ReLU(inplace=True),
                nn.Linear(int(feature_dim * widen), self.fc_out_channels)
            )

    def forward(self, x, target=torch.tensor([0])):
        features = self.head(x)

        if self.loss_mode == 'contrast':
            assert features.size(1) == self.feature_dim
            features = F.normalize(features, dim=1)
            features = features.unsqueeze(1)
            loss = self.criterion(features, target, temperature=0.07)
            features = None
        elif self.loss_mode == 'cross_entropy':
            loss = self.criterion(features, target)
        else:
            raise NotImplementedError

        return loss, features

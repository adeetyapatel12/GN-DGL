import torch
import torch.nn as nn
import math
from utils import div_loss_calc
from ptflops import get_model_complexity_info

torch.manual_seed(0)

class Model_1(nn.Module):

    def __init__(self, avgpool, fc, criterion_ce):
        super(Model_1, self).__init__()
        self.avgpool = avgpool
        self.fc = fc
        self.criterion_ce = criterion_ce

    def forward(self, x, target=torch.tensor([0])):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        loss = self.criterion_ce(logits, target)

class Model_2(nn.Module):
    def __init__(self, conv1, bn1, relu, layer1, layer2, layer3, head, criterion_ce):
        super(Model_2, self).__init__()

        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

        self.head = head
        self.criterion_ce = criterion_ce

    def forward(self, img, target=torch.tensor([0])):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        logits = self.head(x)
        loss = self.criterion_ce(logits, target)

def print_flops(flop_counter):
    print("Maximum flop count of a node: ", max(flop_counter.values()))
    print("Total flop count of all node: ", sum(flop_counter.values()))
    print(flop_counter)

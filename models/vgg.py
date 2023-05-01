'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
from .evonorm import EvoNormSample2d as evonorm_s0
from .utils import ForkedPdb

__all__ = [
    'VGG', 'vgg11'
]

def normalization(planes, groups=2, norm_type='evonorm'):
    if norm_type == "none":
        return nn.Identity()
    elif norm_type == 'batchnorm':
        return nn.BatchNorm2d(planes)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(groups, planes)
    elif norm_type == 'evonorm':
        return evonorm_s0(planes)
    else:
        raise NotImplementedError

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes, dataset):
        super(VGG, self).__init__()
        self.pool     = False
        self.features = features
        if 'imagenet' in dataset:
            self.pool    = True
            self.avgpool = nn.AvgPool2d(7)
        self.classifier  = nn.Sequential(
            nn.Linear(128, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        if self.pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, norm_type, groups):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            norm = normalization(planes=v, groups=groups, norm_type=norm_type)
            if norm_type=='evonorm':
                layers += [conv2d, norm]
            else:
                layers+=[conv2d, norm, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'M': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
}


def vgg11(num_classes, dataset, norm_type, groups=2):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['M'], norm_type, groups), num_classes, dataset)


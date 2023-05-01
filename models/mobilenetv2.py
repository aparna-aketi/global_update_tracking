import torch
import torch.nn as nn
import torch.nn.functional as F
from .evonorm import EvoNormSample2d as evonorm_s0

def normalization(planes, groups=2, norm_type='evonorm'):
    if norm_type == 'batchnorm':
        return nn.BatchNorm2d(planes)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(groups, planes)
    elif norm_type == 'evonorm':
        return evonorm_s0(planes)
    else:
        raise NotImplementedError

class Block_m(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, groups, norm_type):
        super(Block_m, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 =  normalization(planes, groups, norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = normalization(planes, groups, norm_type)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = normalization(out_planes, groups, norm_type)
        self.norm_type = norm_type
        if 'evo' not in self.norm_type:
            self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                normalization(out_planes, groups, norm_type),
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.norm_type!='evonorm':
            out = self.relu(out)
        out = self.bn2(self.conv2(out))
        if self.norm_type!='evonorm':
            out = self.relu(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    #cfg = (expansion, out_planes, num_blocks, stride) 
 

    def __init__(self, num_classes=10, groups = 2, norm_type='evonorm', dataset='imagenette'):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.dataset = dataset
        if 'cifar' in dataset:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.cfg = [(1,  16, 1, 1),
                        (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
                        (6,  32, 3, 2),
                        (6,  64, 4, 2),
                        (6,  96, 3, 1),
                        (6, 160, 3, 2),
                        (6, 320, 1, 1)]
        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.cfg = [(1,  16, 1, 1),
                        (6,  24, 2, 2),  
                        (6,  32, 3, 2),
                        (6,  64, 4, 2),
                        (6,  96, 3, 1),
                        (6, 160, 3, 2),
                        (6, 320, 1, 1)]
        self.bn1 = normalization(32, groups, norm_type)
        self.layers = self._make_layers(in_planes=32, groups=groups, norm_type=norm_type)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = normalization(1280, groups, norm_type)
        self.linear = nn.Linear(1280, num_classes)
        self.norm_type = norm_type
        if 'evo' not in self.norm_type:
            self.relu = nn.ReLU(inplace=True)

    def _make_layers(self, in_planes, groups, norm_type):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block_m(in_planes, out_planes, expansion, stride, groups, norm_type))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if self.norm_type!='evonorm':
            out = self.relu(out)
        out = self.layers(out)
        out = self.bn2(self.conv2(out))
        if self.norm_type!='evonorm':
            out = self.relu(out)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        if 'cifar' in self.dataset:
            out = F.avg_pool2d(out, 4)
        else:
            out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



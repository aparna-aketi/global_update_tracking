import torch
import torch.nn as nn
import math

class EvoNormSample2d(nn.Module):
    def __init__(self, num_features, apply_act=True, groups=8, eps=1e-5, drop_block=None, num_bits=None, num_bits_grad=None):
        super(EvoNormSample2d, self).__init__()
        self.apply_act = apply_act  # apply activation (non-linearity)
        self.groups    = groups
        self.eps       = eps
        param_shape    = (1, num_features, 1, 1)
        self.weight    = nn.Parameter(torch.ones(param_shape), requires_grad=True)
        self.bias      = nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, return_act = False):
        assert x.dim() == 4, 'expected 4D input'
        B, C, H, W = x.shape
        assert C % self.groups == 0
        if self.apply_act:
            hs = torch.clamp((x /6.0) + 0.5, min=0, max=1) #hardsigmoid
            n = x * hs # n = x*hard_sigmoid(x)
            x = x.reshape(B, self.groups, -1)
            x = n.reshape(B, self.groups, -1) / (x.var(dim=-1, unbiased=False, keepdim=True) + self.eps).sqrt()  # x = n/var(x) groupwise instance variance
            x = x.reshape(B, C, H, W)
        if return_act:
            return x * self.weight + self.bias, x    
        return x * self.weight + self.bias # gamma*x+beta

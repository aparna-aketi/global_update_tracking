
from scipy.linalg import orth
import torch
from torch.autograd import Variable
import copy
import numpy as np
from .utils import flatten_tensors, unflatten_tensors, ForkedPdb


class GUT():
    def __init__(self, model, device, rank, learning_rate, averaging_rate=1, neighbors=2, momentum=0, scaling=0.1):
        self.model          = model
        self.rank           = rank
        self.device         = device
        self.avergaing_rate = averaging_rate
        self.lr             = learning_rate
        self.momentum       = momentum
        self.scaling        = scaling
        self.pi             = 1.0/float(neighbors+1)      # !!! this has to updated. right now its hard coded for bidirectional ring topology with uniform weights
        self.y              = []
        self.prev_delta     = []
        self.m_buff         = []
        print("----- Scaling Factor %.2f -------" % self.scaling)
        for params in self.model.module.parameters():
            if params.requires_grad:
                self.y.append(torch.zeros_like(params.data).to(self.device))
                self.prev_delta.append(torch.zeros_like(params.data).to(self.device))
                self.m_buff.append(torch.zeros_like(params.data).to(self.device))

    def __call__(self, global_update, acc_y, lr):
        delta_curr = []
        for p, gu in zip(self.model.module.parameters(), global_update):
            d = copy.deepcopy(p.grad.data)
            d.data.add_(gu.data, alpha = -1.0/self.lr)
            delta_curr.append(d.data)
        
        for y, y_prev, d_curr, d_prev in zip(self.y, acc_y, delta_curr, self.prev_delta):
            y.data.copy_(copy.deepcopy(d_curr.data))
            y.data.add_(y_prev.data, alpha = -self.scaling/self.lr)
            y.data.add_(d_prev.data, alpha = -self.scaling)
            d_prev.data.copy_(copy.deepcopy(d_curr.data))

        self.lr = lr
        return        

   
    def modify_gradients(self):
        """
            Returns
                applies the changes to the model
        """
        ### Applies the grad projections
        count = 0 
        for p in self.model.module.parameters():
            p.grad.data.copy_(self.y[count].data)
            count += 1
        return

        
        
        
        
                
             

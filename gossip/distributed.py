
"""
Distributed Gossip Wrapper
:description: Multi-Threaded Gossip Model Wrapper; designed for efficient
              multi-peer training.
"""

import functools
import time
import sys
import threading
import copy
import os

import torch
import torch.distributed as dist
from torch.cuda.comm import reduce_add_coalesced
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply

from .gossiper import SGD_DS
from .mixing_manager import UniformMixing
from .utils import ( flatten_tensors,
    group_by_dtype, make_logger, unflatten_tensors)

HEARTBEAT_TIMEOUT = 1000  # maximum time to wait for message (seconds)


class GossipDataParallel(Module):
    """ Distributed Gossip model wrapper """

    def __init__(self, module, device_ids=None, rank=None, world_size=None,
                 graph=None, mixing=None, comm_device=None,
                  synch_freq=0, verbose=False, use_streams=False, nesterov=False,
                 eta = 1, momentum=0.9, lr=0.1, weight_decay = 0, neighbors = 2):
        super(GossipDataParallel, self).__init__()

        # devices available locally
        if device_ids is None:
            device_ids     = list(range(torch.cuda.device_count()))
        self.output_device = device_ids[0]
        self.device_ids    = device_ids
        self.lr            = lr
        self.momentum      = momentum
        self.nesterov      = nesterov
        self.weight_decay  = weight_decay
        self.neighbors     = neighbors
    
        if world_size is None or rank is None:
            assert dist.is_initialized()
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        self.process_rank = rank
        
        # put model on output device
        self.module = module
        first_param_dtype = next(self.module.parameters()).dtype
        
        # prepare local intra-node all-reduce objects
        self._module_copies = [self.module]

        # choose communication device based on backend
        if comm_device is None:
            cpu_comm = True if dist.get_backend() == 'gloo' else False
            comm_device = torch.device('cpu') if cpu_comm else torch.device('cuda')
        self.__cpu_comm = comm_device.type == 'cpu'
        #mixing weights for the graph topology
        if mixing is None:
            mixing = UniformMixing(graph, comm_device)
        self.mixing_weights = mixing.get_mixing_weights()
        self.averaging_rate   = torch.ones(1, device=comm_device).type(first_param_dtype)*eta

        # distributed backend config
        self.dist_config = {
            'verbose':      verbose,
            'comm_device':  comm_device,
            'graph':        graph,
            'mixing':       mixing,
            'rank':         rank,
            'process_rank': self.process_rank,
            'world_size':   world_size,
            'cpu_comm':     self.__cpu_comm,
            'epoch':            0.0,
            'iterations':       0}
            

        # logger used to print to stdout
        self.logger = make_logger(rank, verbose)
        self.ps_weight            = torch.ones(1, device=comm_device).type(first_param_dtype)
        self.yi                   = []
        self.yj                   = []
        self.acc_ys               = []
        self.gossip_device_buffer = []
        self.neighbor_copy        = []
        self.self_xhat            = []
        #self.shift_buff           = []
        self.gossip_lock = threading.Lock()
        
        for p in module.parameters():
            cp = p.clone().detach_()
            cp = cp.cpu().pin_memory() if self.__cpu_comm else cp.to(comm_device)
            self.yi.append(copy.deepcopy(cp).to(comm_device))
            self.yj.append(copy.deepcopy(cp).to(comm_device))
            self.acc_ys.append(copy.deepcopy(cp).to(comm_device))
            self.gossip_device_buffer.append(cp)
            self.neighbor_copy.append((1.0-float(self.mixing_weights['self']))*copy.deepcopy(cp).to(comm_device))
            self.self_xhat.append(copy.deepcopy(cp).to(comm_device))
            #self.shift_buff.append(torch.zeros_like(cp).to(comm_device))
    

        if self.dist_config['comm_device'].type != 'cpu' and use_streams:
            self.gossip_stream = torch.cuda.Stream()
        else:
            self.gossip_stream = torch.cuda.current_stream(device=comm_device)

    
        self.gossiper   = SGD_DS (  msg        = flatten_tensors(self.yi), 
                                    device     = self.dist_config['comm_device'],
                                    graph      = self.dist_config['graph'],
                                    mixing     = self.dist_config['mixing'],
                                    rank       = self.dist_config['process_rank'],
                                    world_size = self.dist_config['world_size'],
                                    logger     = self.logger)
        

    def state_dict(self, finish_gossip=True):
        super_dict = super(GossipDataParallel, self).state_dict()
        supplanted_dict = {'state_dict': super_dict,  }
        return supplanted_dict

    def load_state_dict(self, load_dict):
        state_dict = load_dict['state_dict']
        super(GossipDataParallel, self).load_state_dict(state_dict)
       
            
    def forward(self, *inputs, **kwargs):
        """ Forward pass performed in parallel across all devices on node """
        # scatter inputs onto devices
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module(*inputs[0], **kwargs[0])
        
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=0)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs,
                              self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=0)

    def train(self, mode=True):
        super(GossipDataParallel, self).train(mode)
        self.gossip_enable = True
        for module in self._module_copies[1:]:
            module.train(mode)

    def eval(self):
        super(GossipDataParallel, self).eval()
        self.gossip_enable = False
        for module in self._module_copies[1:]:
            module.eval()

    def block(self):
        self.logger.info('blocking')
        dist.barrier()

    def transfer_params(self, mix=True, epoch=0, lr=0.1):
        """ Transfers COPY of model parameters to gossip queue """
        self.lr=lr
        self.dist_config['epoch'] = epoch
        self.dist_config['iterations']+=1
        self.logger.debug('transfering model params')
       
        # compute xi-xi_hat
        for xi, xi_hat, yi in zip(self.module.parameters(), self.self_xhat, self.yi):
            yi.data.copy_(xi.data)
            yi.data.add_(xi_hat.data, alpha=-1)

       
        # gossip during training (not inference)
        send_msg  = flatten_tensors(self.yi)
          
        with self.gossip_lock:
            data_amt = 0
            in_msg, data_amt= self.gossiper.mix(send_msg)

        for r, g in zip(unflatten_tensors(in_msg, self.yj), self.yj):
            if self.dist_config['cpu_comm']:
                g.copy_(r, non_blocking=True)
            else:
                g.data.copy_(r.data)

        #get the avergaed y's of the neighborhood
        for yi, yj, acc_y in zip(self.yi, self.yj, self.acc_ys): 
            acc_y.data.copy_(yj.data)
            acc_y.data.add_(yi.data, alpha = float(self.mixing_weights['self'])) #accumulated y

        #shift the accumulated y
        for xi_hat, xj_hat, acc_y in zip(self.self_xhat, self.neighbor_copy, self.acc_ys): 
            b = 1 + self.momentum
            if self.nesterov: b = b + (self.momentum*self.momentum)
            acc_y.data.add_(xj_hat.data, alpha = 1.0*b)
            acc_y.data.add_(xi_hat.data, alpha = (float(self.mixing_weights['self'])-1.0)*b)

        # update xi_hat
        for xi_hat, yi in zip(self.self_xhat, self.yi):
            xi_hat.data.add_(yi.data)

        #update xj_hats
        for xj_hat, yj in zip(self.neighbor_copy, self.yj): 
            xj_hat.data.add_(yj.data) #update neighbors copy
        
        #get the global update into gossip_device_buffer
        for xj_hat, xi_hat, gossip in zip(self.neighbor_copy, self.self_xhat, self.gossip_device_buffer):
            gossip.data.copy_(xj_hat.data)
            gossip.data.add_(xi_hat.data, alpha = float(self.mixing_weights['self'])-1.0) #transfer global update to gossip_buffer

        self.gossip_stream.synchronize()
        return True, data_amt, copy.deepcopy(self.gossip_device_buffer), self.acc_ys

    
    def gossip_averaging(self):
        #gossip averaging step   
        for p, r in zip(self.module.parameters(), self.gossip_device_buffer):
            r.data.mul_(self.averaging_rate.type(r.data.dtype))
            p.data.add_(r.data)  
        return
    
    def remove_gossip(self):
        for p, r in zip(self.module.parameters(), self.gossip_device_buffer):
            r.data.mul_(self.averaging_rate.type(r.data.dtype))
            p.data.add_(r.data, alpha=-1)  
        return

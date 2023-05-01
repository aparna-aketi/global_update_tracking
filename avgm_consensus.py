import torch
import random
import numpy as np
import torch.nn.functional as F
import copy
from matplotlib import pyplot as plt
import matplotlib
import wandb


n        = 64
T        = 1000
x_size   = 200000
scaling  = 0.1
scalingm = 0.01
W      = torch.zeros(n,n).to('cuda')
I      = torch.eye(n).to("cuda")

W[0,0] = W[n-1,n-1] = 1.0/3.0
W[0,1] = W[0,n-1] = W[n-1,0] = W[n-1,n-2] = 1.0/3.0
for i in range(1,n-1):
    for j in [i-1, i, i+1]:
        if i==j: W[i,j] = 1.0/3.0
        else: W[i,j] = 1.0/3.0

#print(W, W1)


Error_sg   = []
Error_gut  = []
Error_qgm  = []
Error_gutm = []
#print(W.size(), x_init.size())
for i in range(0,2):
    x_init  = -2 * torch.rand(x_size, n) + 1.0
    x_init  =  x_init.to('cuda')
    x_final = torch.mean(x_init, dim=1).view(-1,1)
    x_final = x_final.expand(x_size, n)

    #no momentum
    x_t    = copy.deepcopy(x_init)
    x_g    = copy.deepcopy(x_init)
    x_prev = copy.deepcopy(x_init)
    d_prev = torch.zeros_like(x_prev)
    #momentum
    x_q        = copy.deepcopy(x_init)
    x_gm       = copy.deepcopy(x_init)
    x_prevm    = copy.deepcopy(x_init)
    d_prevm    = torch.zeros_like(x_prevm)
    m_buff     = torch.zeros_like(x_prevm)
    mg_buff    = torch.zeros_like(x_prevm)
    shift      = torch.zeros_like(x_prevm)
    error_sg  = []
    error_gut = []
    error_qgm  = []
    error_gutm = []
    Time      = []
    for t in range(0,T):
        #simple gossip update
        d = torch.matmul(x_t, W-I)
        x_t.data.add_(d.data)
        ####
        #update using tracking method (GUT)
        dg = torch.matmul(x_g, W-I)
        dg_copy   = copy.deepcopy(dg)
        xg_copy = copy.deepcopy(x_g)
        bias = torch.matmul(x_g - x_prev, W) - d_prev + torch.matmul(x_g, W-I)
        x_g.data.add_(dg.data)
        x_g.data.add_(bias.data, alpha=scaling)
        d_prev.data.copy_(dg_copy.data)
        x_prev.data.copy_(xg_copy.data)
        #gossip update with qgm
        dm = torch.matmul(x_q, W-I)
        dm.data.add_(torch.matmul(m_buff, W), alpha=0.9)
        x_q.data.add_(dm.data)
        m_buff.data.mul_(0.9).add_(dm.data, alpha=0.1)
        ####
        #update using tracking method with qgm (GUTm)
        dg        = torch.matmul(x_gm, W-I)
        dg_copy   = copy.deepcopy(dg)
        xg_copy   = copy.deepcopy(x_gm)
        #shift.data.mul_(0.9).add_(torch.matmul(x_gm, (W-I)), alpha=0.1)
        bias      = torch.matmul(x_gm - x_prevm, W) - d_prevm + torch.matmul(x_gm, (W-I))
        u         = dg.data + (scalingm*bias.data)
        u.data.add_(torch.matmul(mg_buff, W), alpha=0.9)
        x_gm.data.add_(u.data)
        mg_buff.data.mul_(0.9).add_(u.data, alpha=0.1)
        d_prevm.data.copy_(dg_copy.data)
        x_prevm.data.copy_(xg_copy.data)
        
        #compute the error without momentum
        x_copy = copy.deepcopy(x_t)
        xg_copy = copy.deepcopy(x_g)
        x_copy.data.add_(x_final.data, alpha=-1.0)
        xg_copy.data.add_(x_final.data, alpha=-1.0)
        x_copy.square_()
        xg_copy.square_()
        e  = (1.0/n)*torch.sum(x_copy.data)
        e1 = (1.0/n)*torch.sum(xg_copy.data)
        #print("Average consensus error at iteration %d: %.4f, %.4f"%(t, e, e1))
        error_sg.append(e.cpu().numpy())
        error_gut.append(e1.cpu().numpy())
        #compute the error with momentum
        x_copy = copy.deepcopy(x_q)
        xg_copy = copy.deepcopy(x_gm)
        x_copy.data.add_(x_final.data, alpha=-1.0)
        xg_copy.data.add_(x_final.data, alpha=-1.0)
        x_copy.square_()
        xg_copy.square_()
        e  = (1.0/n)*torch.sum(x_copy.data)
        e1 = (1.0/n)*torch.sum(xg_copy.data)
        #print("Average consensus error at iteration %d: %.4f, %.4f"%(t, e, e1))
        error_qgm.append(e.cpu().numpy())
        error_gutm.append(e1.cpu().numpy())
        Time.append(t)
    Error_sg.append(error_sg)
    Error_gut.append(error_gut)
    Error_qgm.append(error_qgm)
    Error_gutm.append(error_gutm)
    

Error_sg       = np.array(Error_sg)
Error_gut      = np.array(Error_gut)
Error_sg_mean  = np.mean(Error_sg, axis=0)
Error_gut_mean = np.mean(Error_gut, axis=0)
Error_sg_std   = np.std(Error_sg, axis=0)
Error_gut_std  = np.std(Error_gut, axis=0)
Error_qgm       = np.array(Error_qgm)
Error_gutm      = np.array(Error_gutm)
Error_qgm_mean  = np.mean(Error_qgm, axis=0)
Error_gutm_mean = np.mean(Error_gutm, axis=0)
Error_qgm_std   = np.std(Error_qgm, axis=0)
Error_gutm_std  = np.std(Error_gutm, axis=0)


fig,ax = plt.subplots()
#fig.set_size_inches(10,8)
plt.box(False)
ax.plot(Time, Error_sg_mean, label = "simple gossip", linewidth=1.2, color='red', linestyle='dashed')
ax.fill_between(Time, (Error_sg_mean-Error_sg_std), (Error_sg_mean+Error_sg_std), alpha=0.1, color='red')
ax.plot(Time, Error_gut_mean, label = "gossip with GUT", linewidth=1.2, color='dodgerblue', linestyle='dashed')
ax.fill_between(Time, (Error_gut_mean-Error_gut_std), (Error_gut_mean+Error_gut_std), alpha=0.1, color='dodgerblue')
ax.plot(Time, Error_qgm_mean, label = "gossip with QGM", linewidth=1.2, color='red')
ax.fill_between(Time, (Error_qgm_mean-Error_qgm_std), (Error_qgm_mean+Error_qgm_std), alpha=0.1, color='red')
ax.plot(Time, Error_gutm_mean, label = "gossip with QG-GUTm", linewidth=1.2, color='dodgerblue')
ax.fill_between(Time, (Error_gutm_mean-Error_gutm_std), (Error_gutm_mean+Error_gutm_std), alpha=0.1, color='dodgerblue')
#ax.plot(Time, Error_gut_mean, label = "gossip with GUT")
#fig.patch.set_visible(False)
matplotlib.pyplot.yscale("log")
matplotlib.pyplot.grid(True, color='0.8')
ax.set_xlabel("Iteration", fontsize=18)
ax.set_ylabel("Consensus Error", fontsize=18)
#ax.set_title("Average consensus on 64 nodes ring")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax.set_xticks([0, 250, 500, 750, 1000])
plt.tight_layout()
plt.legend(fontsize=13)
plt.savefig("ac_64.pdf")
plt.show()
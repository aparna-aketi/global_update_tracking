import argparse
import os
import shutil
import time
from   tkinter import E
import numpy as np
import copy
import wandb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from   torchsummary import summary
import torch.nn.functional as F
import random


# Importing modules related to distributed processing
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.autograd import Variable
from torch.multiprocessing import spawn
from tensorboardX import SummaryWriter

###########
from gossip import GossipDataParallel
from gossip import RingGraph, GridGraph, FullGraph, PetersenGraph, DyckGraph
from gossip import UniformMixing
from gossip import *
from models import *
from partition_data import *

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet', help = 'resnet or vgg or resquant' )
parser.add_argument('-depth', '--depth', default=20, type=int, help='depth of the resnet model')
parser.add_argument('--normtype',   default='evonorm', help = 'none or batchnorm or groupnorm or evonorm' )
parser.add_argument('--data-dir', dest='data_dir',    help='The directory used to save the trained models',   default='../../data', type=str)
parser.add_argument('--dataset', dest='dataset', type=str,     help='available datasets: cifar10, cifar100, imagenette, fmnist', default='cifar10')
parser.add_argument('--skew', default=1.0, type=float,     help='parameter alpha that controls non-iidness')
parser.add_argument('--classes', default=10, type=int,     help='number of classes in the dataset')
parser.add_argument('-b', '--batch-size', default=512, type=int,  help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,     metavar='LR', help='initial learning rate')
parser.add_argument('--gamma',  default=1.0, type=float,  metavar='AR', help='averaging rate')
parser.add_argument('--momentum', default=0.0, type=float, metavar='M',     help='momentum')
parser.add_argument('-world_size', '--world_size', default=5, type=int, help='total number of nodes')
parser.add_argument('--epochs', default=200, type=int, metavar='N',   help='number of total epochs to run')
parser.add_argument('--scaling',  default=0.9, type=float, help='scaling factor for the bias correction term')
parser.add_argument('--graph', '-g',  default='ring', help = 'graph structure - [ring, torus, dyck, peterson, full]' )
parser.add_argument('--neighbors', default=2, type=int,     help='number of neighbors per node')
parser.add_argument('-d', '--devices', default=4, type=int, help='number of gpus/devices on the card')
parser.add_argument('-j', '--workers', default=4, type=int,  help='number of data loading workers (default: 4)')
parser.add_argument('--seed', default=1234, type=int,   help='set seed')
parser.add_argument('--print-freq', '-p', default=100, type=int,  help='print frequency (default: 50)')
parser.add_argument('--save-dir', dest='save_dir',    help='The directory used to save the trained models',   default='outputs', type=str)
parser.add_argument('--port', dest='port',   help='between 3000 to 65000',default='25500' , type=str)
parser.add_argument('--save-every', dest='save_every',  help='Saves checkpoints at every specified number of epochs',  type=int, default=5)
parser.add_argument('--nesterov', action='store_true', )
args = parser.parse_args()

# Check the save_dir exists or not
args.save_dir = os.path.join(args.save_dir, args.arch+"_nodes_"+str(args.world_size)+"_"+ args.normtype+"_lr_"+ str(args.lr)+"_seed_"+str(args.seed)+"_skew_"+str(args.skew)+"_"+args.graph )
if not os.path.exists(os.path.join(args.save_dir, "excel_data") ):
    os.makedirs(os.path.join(args.save_dir, "excel_data") )
torch.save(args, os.path.join(args.save_dir, "training_args.bin"))    

    
def partition_trainDataset(device):
    """Partitioning dataset""" 
    if args.dataset == 'cifar10':
        normalize   = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        classes    = 10
        class_size = {x:5000 for x in range(10)}

        dataset = datasets.CIFAR10(root=args.data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
    elif args.dataset == 'cifar100':
        normalize  = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
        classes    = 100
        class_size = {x:500 for x in range(100)}

        dataset = datasets.CIFAR100(root=args.data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
    elif args.dataset == 'fmnist':
        normalize  = transforms.Normalize((0.5,), (0.5,))
        classes    = 10
        class_size = {x:6000 for x in range(10)}

        dataset = datasets.FashionMNIST(root=args.data_dir, train = True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)
    elif args.dataset == 'imagenette':
        normalize  = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        classes    = 10
        class_size = {0: 963, 1: 955, 2: 993, 3: 858, 4: 941, 5: 956, 6: 961, 7: 931, 8: 951, 9: 960}

        data_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(), normalize,])

        data_dir = args.data_dir

        dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)
                  
       
    size = dist.get_world_size()
    #print(size)
    bsz = int((args.batch_size) / float(size))
    
    partition_sizes = [1.0/size for _ in range(size)]
    partition = DataPartitioner(args.seed, dataset, partition_sizes, non_iid_alpha=args.skew, partition_type="non_iid_dirichlet")

    partition, data_distribution = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True, num_workers=2)
    return train_set, bsz, data_distribution

def test_Dataset():
    if args.dataset=='cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
        dataset = datasets.CIFAR10(root=args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset=='cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
        dataset = datasets.CIFAR100(root=args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset=='fmnist':
        normalize = transforms.Normalize((0.5,), (0.5,))
        dataset   = datasets.FashionMNIST(root=args.data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset == 'imagenette':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        data_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(), normalize,])
        data_dir = args.data_dir
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),  data_transforms)

    val_bsz = 128
    val_set = torch.utils.data.DataLoader(dataset, batch_size=val_bsz, shuffle=False, num_workers=2)
    return val_set, val_bsz



def run(rank, size):
    global args, best_prec1, global_steps
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:{}".format(rank%args.devices))
	##############
    best_prec1 = 0
    data_transferred = 0
    
    if args.arch.lower()=='resnet':
        model = resnet(num_classes=args.classes, depth=args.depth, dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'vgg11':
        model = vgg11(num_classes=args.classes, dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'mobilenet':
        model = MobileNetV2(num_classes=args.classes, dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'lenet5':
        model = LeNet5()
    else:
        raise NotImplementedError
    
    if rank==0:
        print(args)
        print('Printing model summary...')
        if args.dataset=="fmnist":
            print(summary(model, (1,28,28), batch_size=int(args.batch_size/size), device='cpu'))
        elif args.dataset=="imagenette":
            print(summary(model, (3, 224, 224), batch_size=int(args.batch_size/size), device='cpu'))
        else: 
            print(summary(model, (3, 32, 32), batch_size=int(args.batch_size/size), device='cpu'))
        
    if args.graph.lower()   == 'ring':
        graph = RingGraph(rank, size, args.devices, peers_per_itr=args.neighbors) #undirected ring structure => neighbors = 2 ; directed ring => neighbors=1
    elif args.graph.lower() == 'torus':   
        graph = GridGraph(rank, size, args.devices, peers_per_itr=args.neighbors) # torus graph structure
    elif args.graph.lower() == 'petersen':   
        graph = PetersenGraph(rank, size, args.devices, peers_per_itr=3) # petersen graph structure -- cubic graph
    elif args.graph.lower() == 'dyck':   
        graph = DyckGraph(rank, size, args.devices, peers_per_itr=3) # dyck graph structure -- cubic graph
    elif args.graph.lower() == 'full':
        graph = FullGraph(rank, size, args.devices, peers_per_itr=args.world_size-1) # torus graph structure  
    else:
        raise NotImplementedError
    #graph = BipartiteGraph(rank, size, args.devices, peers_per_itr=int(args.world_size/2)) #undirected bipartite structure, use only for even world size
   
    mixing = UniformMixing(graph, device)
    model = GossipDataParallel(model, 
				device_ids=[rank%args.devices],
				rank=rank,
				world_size=size,
				graph=graph, 
				mixing=mixing,
				comm_device=device,
                eta = args.gamma, 
                momentum=args.momentum,
                nesterov=args.nesterov,
                weight_decay=0.0,
                lr = args.lr, 
                neighbors = args.neighbors) 
    model.to(device)
    train_loader, bsz_train, _ = partition_trainDataset(device=device)
    val_loader, bsz_val     = test_Dataset()

    # define loss function (criterion) and nvidia-smi optimizer
    tracker   = GUT(model, device, rank, args.lr, args.gamma, neighbors=args.neighbors, momentum = args.momentum, scaling = args.scaling)
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, nesterov=args.nesterov)
    
    if rank==0: print(optimizer)
    criterion    = nn.CrossEntropyLoss().to(device)
    milestones   = [int(args.epochs*0.5), int(args.epochs*0.75)]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma = 0.1, milestones=milestones)

    lr            = optimizer.param_groups[0]['lr']
    for epoch in range(0, args.epochs):  
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        model.block()
        dt, train_acc, train_loss, lr = train(train_loader, model, criterion, optimizer, epoch, lr, device, tracker)
        data_transferred += dt
        lr_scheduler.step()
        prec1, loss = validate(val_loader, model, criterion, bsz_val,device, epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, filename=os.path.join(args.save_dir, 'model_{}.th'.format(rank)))
      
    #############################
    dt = gossip_avg(train_loader, model, criterion, optimizer, epoch, optimizer.param_groups[0]['lr'], device)
    print('Final test accuracy')
    prec1_final, _ = validate(val_loader, model, criterion, bsz_val,device, epoch, False, args.classes, return_classwise=False)
    print("Rank : ", rank, "Data transferred(in GB) during training: ", data_transferred/1.0e9, "Data transferred(in GB) in final gossip averaging rounds: ", dt/1.0e9, "\n")
    #Store processed data
    torch.save((prec1, prec1_final, (data_transferred+dt)/1.0e9), os.path.join(args.save_dir, "excel_data","rank_{}.sp".format(rank)))


def train(train_loader, model, criterion, optimizer, epoch, lr, device, tracker=None):
    """
        Run one train epoch
    """
    batch_time       = AverageMeter()
    data_time        = AverageMeter()
    losses           = AverageMeter()
    top1             = AverageMeter()
    data_transferred = 0

    # switch to train mode
    model.train()
    end = time.time()
    step = len(train_loader)*epoch
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_var, target_var = Variable(input).to(device), Variable(target).to(device)
        # gossip the weights 
        _, amt_data_transfer, global_update, acc_y  = model.transfer_params(epoch=epoch+(1e-3*i), lr=optimizer.param_groups[0]['lr'])
        data_transferred                           += amt_data_transfer
        #add gossip to compute gradients
        model.gossip_averaging()
        # compute output in the forward pass
        output = model(input_var)
        #compute loss
        loss = criterion(output, target_var)
        # compute gradient 
        loss.backward()
        #remove gossip as it is added directly to the gradients in the tracker function
        model.remove_gossip()
        #update tracking variable
        tracker(global_update, acc_y, optimizer.param_groups[0]['lr'])
        tracker.modify_gradients()
        # do local update
        optimizer.step()
        #zero out the gradients
        optimizer.zero_grad() 
        #model.gossip_averaging()
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Rank: {0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      dist.get_rank(), epoch, i, len(train_loader),  batch_time=batch_time,
                      loss=losses, top1=top1))
        step += 1 
    return data_transferred, top1.avg, losses.avg, lr

def gossip_avg(train_loader, model, criterion, optimizer, epoch, lr, device):
    """
       This function runs only gossip averaging for 50 iterations without local sgd updates - used to obtain the average model
    """
    data_transferred = 0 
    n = 50
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input_var, target_var = Variable(input).to(device), Variable(target).to(device)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.zero_grad()
        _, amt_data_transfer, _, _ = model.transfer_params(epoch=epoch+(1e-3*i), lr=lr)
        model.gossip_averaging()
        data_transferred += amt_data_transfer
        if i==n: break
    return data_transferred

def validate(val_loader, model, criterion, batch_size, device, epoch=0, class_wise=False, list_of_classes=10, return_classwise=False):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    step = len(val_loader)*epoch
    acc         = [0 for c in range(list_of_classes)]
    class_count = [0 for c in range(list_of_classes)]

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var, target_var = Variable(input).to(device), Variable(target).to(device)

            # compute output and loss
            output = model(input_var)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if class_wise:
                _, preds = torch.max(output.data, 1)
                for c in range(list_of_classes):
                    acc[c] += ((preds == target_var) * (target_var == c)).sum().float() 
                    class_count[c] += (target_var == c).sum()
                

            if i % args.print_freq == 0:
                print('Rank: {0}\t'
                      'Test: [{1}/{2}]\t'
                      #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          dist.get_rank(),i, len(val_loader), 
                          #batch_time=batch_time, 
                          loss=losses,
                          top1=top1))
            step += 1
    print('Rank:{0}, Prec@1 {top1.avg:.3f}'.format(dist.get_rank(),top1=top1))
    if class_wise:
        for c in range(list_of_classes):
            acc[c] = (acc[c].cpu().numpy()/class_count[c].cpu().numpy())*100
        print('Class-wise accuracy for rank {} is '.format(dist.get_rank()), acc)
    if return_classwise:
        return top1.avg, acc
    return top1.avg, losses.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def flatten_tensors(tensors):
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)
    return flat

def init_process(rank, size, fn, backend='nccl'):
    """Initialize distributed enviornment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank,size)



if __name__ == '__main__':
    size = args.world_size
    spawn(init_process, args=(size, run), nprocs=size, join=True)
    #read stored data
    excel_data = {
        'data': args.dataset,
        'arch': args.arch,
        "momentum":args.momentum,
        "nesterov":args.nesterov,
        "learning rate": args.lr,
        "gamma" : args.gamma,
        "graph" : args.graph,
        "skew" : args.skew,
        "norm" : args.normtype,
        "epochs": args.epochs,
        "nodes": size,
        "avg test acc":[0.0 for _ in range(size)],
        "avg test acc final":[0.0 for _ in range(size)],
        "data transferred": [0.0 for _ in range(size)],
        "seed" :args.seed,
        "scaling": args.scaling,
        'depth':args.depth,
         }
    for i in range(size):
        acc, acc_final, d_tfr = torch.load(os.path.join( args.save_dir, "excel_data","rank_{}.sp".format(i) ))
        excel_data["avg test acc"][i] = acc
        excel_data["avg test acc final"][i] = acc_final
        excel_data["data transferred"][i] = d_tfr
    torch.save(excel_data, os.path.join(args.save_dir, "excel_data","dict"))
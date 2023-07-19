#!/usr/bin/env python

print("Start at the beginning of training!")
import io, os, sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
import torch.utils.data.distributed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import argparse

# For memory debug!
#import psutil
#process = psutil.Process(os.getpid())

#----------------------Parser settings---------------------------

parser = argparse.ArgumentParser(description='Training_DDP')

parser.add_argument('--batch_size',     type=int,   default=512,
                    help='input batch size for training (default: 512)')
parser.add_argument('--epochs',         type=int,   default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr',             type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--seed',           type=int,   default=42,
                    help='random seed (default: 42)')
parser.add_argument('--log-interval',   type=int,   default=10, 
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--device',         default='gpu', choices=['cpu', 'gpu'],
                    help='Whether this is running on cpu or gpu')
parser.add_argument('--shuffle',        default='global', choices=['local', 'global'],
                    help='Whether doing global shuffle or local shuffle')
parser.add_argument('--phase',          type=int,   default=0,
                    help='the current phase of workflow, phase0 will not read model')
parser.add_argument('--num_threads',    type=int,   default=0, 
                    help='set number of threads per worker. only work for cpu')
parser.add_argument('--num_workers',    type=int,   default=1, 
                    help='set the number of op workers. only work for gpu')

args = parser.parse_args()
args.cuda = ( args.device.find("gpu")!=-1 and torch.cuda.is_available() )


#--------------------DDP initialization-------------------------

size = int(os.getenv("SLURM_NPROCS"))
rank = int(os.getenv("SLURM_PROCID"))
local_rank = int(os.getenv("SLURM_LOCALID"))

fn = 'output_' + str(rank) + '.txt'
with open(fn, 'a') as f:
    print("DDP: I am worker size = {}, rank = {}, local_rank = {}".format(size, rank, local_rank), file=f)

# Pytorch will look for these:
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(size)

if args.device == "gpu": backend = 'nccl'
elif args.device == "cpu": backend = 'gloo'

with open(fn, 'a') as f:
    print(args)
    print(backend)

torch.distributed.init_process_group(backend=backend, init_method='file:///pscratch/sd/t/tianle/sharedfile', world_size=size, rank=rank)
with open(fn, 'a') as f:
    print("rank = {}, is_initialized = {}, nccl_avail = {}, get_rank = {}, get_size = {}".format(rank, torch.distributed.is_initialized(), torch.distributed.is_nccl_available(), torch.distributed.get_rank(), torch.distributed.get_world_size()), file=f)

if args.cuda:
    # DDP: pin GPU to local rank.
    with open(fn, 'a') as f:
        print("rank = {}, local_rank = {}, num_of_gpus = {}".format(rank, local_rank, torch.cuda.device_count()), file=f)

    # Handles the case where we pinned GPU to local rank in run script
    if torch.cuda.device_count() == 1:
        torch.cuda.set_device(0)
    else:
        torch.cuda.set_device(int(local_rank))
#        torch.cuda.set_device(torch.cuda.device_count() - 1 - int(local_rank)) # handles Polaris NUMA topology
    torch.cuda.manual_seed(args.seed + int(rank))

if (not args.cuda) and (args.num_threads!=0):
    torch.set_num_threads(args.num_threads)

with open(fn, 'a') as f:
    print("Rank = {}".format(rank), " Torch Thread setup with number of threads: ", torch.get_num_threads(), " with number of inter_op threads: ", torch.get_num_interop_threads(), file=f)

torch.manual_seed(args.seed + int(rank))


if args.shuffle == 'global':
    X_scaled = torch.tensor(1.0 * np.arange(0, size * 100))
    Y_scaled = torch.tensor(2.0 * np.arange(0, size * 100))
    with open(fn, 'a') as f:
        print("X = ", X_scaled, file=f)
        print("Y = ", Y_scaled, file=f)
    
    #----------------DDP: use DistributedSampler to partition the train/test data--------------------
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    train_dataset = torch.utils.data.TensorDataset(X_scaled, Y_scaled)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
elif args.shuffle == 'local':   # Use keweis' suggestion to completely get rid of sampler for local shuffle
    X_scaled = torch.tensor(1.0 * np.arange(rank * 100, (rank + 1) * 100))
    Y_scaled = torch.tensor(2.0 * np.arange(rank * 100, (rank + 1) * 100))
    with open(fn, 'a') as f:
        print("X = ", X_scaled, file=f)
        print("Y = ", Y_scaled, file=f)
    
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    train_dataset = torch.utils.data.TensorDataset(X_scaled, Y_scaled)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

def metric_average(val, name):
    # Sum everything and divide by total size:
    dist.all_reduce(val,op=dist.reduce_op.SUM)
    val /= size
    return val

#------------------------------start training----------------------------------

def train(epoch):
    if args.shuffle == 'global':
        train_sampler.set_epoch(epoch)
    with open(fn, 'a') as f:

        for batch_idx, current_batch in enumerate(train_loader):     
            if args.cuda:
                inp, current_batch_y = current_batch[0].cuda(), current_batch[1].cuda()
            else:
                inp, current_batch_y = current_batch[0],        current_batch[1]
            print(batch_idx, file=f)
            print(current_batch, file=f)
    
for epoch in range(1, args.epochs + 1):
    with open(fn, 'a') as f:
        print("epoch = ", epoch, file=f)
    train(epoch)

torch.distributed.destroy_process_group()

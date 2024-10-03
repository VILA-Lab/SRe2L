'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''
# ------------------------------------------------------------------------------
# Multi-GPU training code for recover in SRe2L is modified from https://github.com/VILA-Lab/SRe2L 
# This version is revised by Xiaochen Ma (https://ma.xiaochen.world/)
# ------------------------------------------------------------------------------
import time
import torch
import numpy as np
import datetime
import builtins
import torch.distributed as dist
from collections import defaultdict, deque

def distributed_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def distribute_setup(rank, world_size, args):
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    builtin_print = builtins.print

    is_master = rank == 0

    # hack print to add timestamp to logger.
    builtin_print("=============================================\n",
                  "   Print is hacked, only rank 0 can print\n",
                 f"   Last message from Rank {rank}\n",
                  "============================================="
                  )
    builtin_print()
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (world_size > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)
    builtins.print = print


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def beta_policy(mom_fn):
    def _alr(optimizer, iteration, epoch, param, indx):
        mom = mom_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group[param][indx] = mom

    return _alr


def mom_cosine_policy(base_beta, warmup_length, epochs):
    def _beta_fn(iteration, epoch):
        if epoch < warmup_length:
            beta = base_beta * (epoch + 1) / warmup_length
        else:
            beta = base_beta
        return beta

    return beta_policy(_beta_fn)


def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    dtype = torch.float16 if use_fp16 else torch.float32


    mean = torch.tensor([0.485, 0.456, 0.406], dtype=dtype, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=dtype,device=image_tensor.device).view(1, 3, 1, 1)

    min_vals = - mean / std
    max_vals = (1 - mean) / std
    
    clipped = torch.clamp(image_tensor, min_vals, max_vals)
    return clipped


def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    dtype = torch.float16 if use_fp16 else torch.float32
    
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=dtype, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=dtype, device=image_tensor.device).view(1, 3, 1, 1)
    
    denormalized = image_tensor * std + mean
    
    denormalized = torch.clamp(denormalized, 0.0, 1.0)
    
    return denormalized

class BNFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()

def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0

    return loss_var_l1, loss_var_l2

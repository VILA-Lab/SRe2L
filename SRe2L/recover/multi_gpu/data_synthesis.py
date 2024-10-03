'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

# ------------------------------------------------------------------------------
# Multi-GPU training code for recover in SRe2L is modified from https://github.com/VILA-Lab/SRe2L 
# This version is revised by Xiaochen Ma (https://ma.xiaochen.world/)
# ------------------------------------------------------------------------------
import os
import random
import datetime
import builtins
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.distributed as dist
import torch.utils.data.distributed
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from utils import *

from data_utils import MultiCardDataset
from torch.utils.data import DataLoader, DistributedSampler

def parse_args():
    parser = argparse.ArgumentParser(
        "SRe2L: recover data from pre-trained model")
    
    """Data save flags"""
    parser.add_argument('--exp-name', type=str, default='test',
                        help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--syn-data-path', type=str,
                        default='./syn_data', help='where to store synthetic data')
    parser.add_argument('--print_period', type=int,
                        default=100, help='print period, count by iteration')
    parser.add_argument('--store-best-images', action='store_true',
                        help='whether to store best images')
    
    """Optimization related flags"""
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--image_shifting', default=32, type=int, help='random shift on the synthetic data')
    parser.add_argument('--r-bn', type=float, default=0.05,
                        help='coefficient for BN feature distribution regularization')
    parser.add_argument('--first-bn-multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_bn')
    parser.add_argument('--tv-l2', type=float, default=0.0001,
                        help='coefficient for total variation L2 loss')
    parser.add_argument('--l2-scale', type=float,
                        default=0.00001, help='l2 loss on the image')
    
    """Model related flags"""
    parser.add_argument('--arch-name', type=str, default='resnet18',
                        help='arch name from pretrained torchvision models')
    parser.add_argument('--verifier', action='store_true',
                        help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='mobilenet_v2',
                        help="arch name from torchvision models to act as a verifier")
    """IPC (Image Per Class) flags"""
    parser.add_argument("--ipc", default=50, type=int, help="IPC, Image Per Class")

    parser.add_argument('--num-classes', type=int, default=1000,
                        help='number of classes in the dataset')
    
    """Distributed training parameters"""
    parser.add_argument('--world-size', type=int, default=1,
                        help='number of distributed processes/gpus')
    parser.add_argument('--dist-url', type=str, default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                        help='distributed backend')
    
    args = parser.parse_args()

    args.syn_data_path= os.path.join(args.syn_data_path, args.exp_name)
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)
    return args


def get_images(args, model_teacher, func_for_validate, rank, world_size):
    print(f"Rank {rank}: get_images call for IPC {args.ipc}")

    # list of hooks for BN layers
    loss_r_feature_layers = []
    
    # add hooks for BN layers, to calculate BN feature loss
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module))
            
    model_teacher = model_teacher.to(rank)

    # data loader, only load the class_id & saving path
    dataset = MultiCardDataset(num_classes=args.num_classes, ipc=args.ipc, root_dir=args.syn_data_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False)

    # loop over the different samples
    for batch in dataloader:
        targets = batch['class_id'].to(rank)
        path_all = batch['path']
        
        # generate random noise image as input
        inputs = torch.randn(
            (targets.shape[0], 3, 224, 224), 
            requires_grad=True, 
            device='cuda',
            dtype=torch.float
        )
        inputs = inputs.to(rank)

        # magic number 迭代多少iter去生成一张图
        iterations_per_layer = args.iteration
        lim_0, lim_1 = args.image_shifting , args.image_shifting # random shift limits default 32

        # Adam optimizer
        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps = 1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer) # 0 - do not use warmup
        # loss function
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        
        # loop over the selected noise image to optimize it
        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)
            # Augmentation
            aug_function = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ])
            inputs_jit = aug_function(inputs).to(rank) # ？ 边缘区域欠优化？

            # apply random jitter offsets
            off1 = random.randint(0, lim_0)
            off2 = random.randint(0, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()
            # print(rank, model_teacher.device)
            outputs = model_teacher(inputs_jit)

            # R_cross classification loss
            loss_ce = criterion(outputs, targets)

            # R_feature loss ’first_bn_multiplier‘ = 10
            rescale = [args.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
            loss_r_bn_feature = sum([bn_hooks.r_feature * rescale[idx] for (idx, bn_hooks) in enumerate(loss_r_feature_layers)])

            # R_prior losses
            _, loss_var_l2 = get_image_prior_losses(inputs_jit)

            # l2 loss on images
            loss_l2 = torch.norm(inputs_jit.reshape(args.batch_size, -1), dim=1).mean()

            # combining losses
            loss_aux = args.tv_l2 * loss_var_l2 + \
                        args.l2_scale * loss_l2 + \
                        args.r_bn * loss_r_bn_feature

            # combine classification loss and smoothing losses
            loss = loss_ce + loss_aux
            if iteration % args.print_period==0:
                print("----------RANK {}  iteration {} ----------".format(rank, iteration))
                print("total loss", loss.item())
                print("target slice start:", targets[:10])
                print("target slice ends:", targets[-10:])
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("main criterion", criterion(outputs, targets).item())
                # comment below line can speed up the training (no validation process)
                if func_for_validate is not None:
                    func_for_validate(inputs, targets)

            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            inputs.data = clip(inputs.data)

        if args.store_best_images:
            best_inputs = inputs.data.clone() # using multicrop, save the last one
            best_inputs = denormalize(best_inputs)
            for id in range(best_inputs.shape[0]):

                # save into separate folders
                dir_path = os.path.dirname(path_all[id])
                place_to_store = path_all[id]
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                image_np = best_inputs[id].data.cpu().numpy().transpose((1, 2, 0))
                pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
                pil_image.save(place_to_store)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
    torch.cuda.empty_cache()

def validate(input, target, model):
    def accuracy(output:torch.Tensor, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())

def main_syn(args, rank, world_size):
    # 读取模型
    model_teacher = models.__dict__[args.arch_name](weights='IMAGENET1K_V1')
    # eval模式，关掉dropout，batchnorm等
    model_teacher.eval()
    # 不更新梯度
    for p in model_teacher.parameters():
        p.requires_grad = False
    
    # New API for loading models
    # https://pytorch.org/vision/stable/models.html
    model_verifier = models.__dict__[args.verifier_arch](weights='IMAGENET1K_V1')
    model_verifier = model_verifier.to(rank)
    model_verifier.eval()
    for p in model_verifier.parameters():
        p.requires_grad = False

    function_for_validate = lambda x,y: validate(x, y, model_verifier)
    
    # create folder for synthetic data
    if not os.path.exists(args.syn_data_path):
        os.mkdir(args.syn_data_path)
    
    get_images(args, model_teacher, function_for_validate, rank, world_size)
   
if __name__ == '__main__':
    args = parse_args()
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['LOCAL_RANK'])  # torchrun 会设置 LOCAL_RANK 环境变量

    # init distributed
    distribute_setup(rank, world_size, args)
    try:
        print(f'Rank {rank} is running')
        main_syn(args, rank, world_size)
    finally:
        # clean up the distributed training
        dist.destroy_process_group()

import os
import random
import argparse
import collections
import numpy as np
from PIL import Image
import time

import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision 

from utils import *

my_device = 'cuda:0'

def get_images(args, model_teacher, hook_for_display, ipc_id):
    print("get_images call")
    save_every = 100
    batch_size = args.batch_size

    best_cost = 1e4

    loss_r_feature_layers = []
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module))

    targets_all = torch.LongTensor(np.arange(1000))

    for kk in range(0, 1000, batch_size):
        targets = targets_all[kk:min(kk+batch_size,1000)].to(my_device)

        data_type = torch.float
        inputs = torch.randn((targets.shape[0], 3, 224, 224), requires_grad=True, device=my_device,
                             dtype=data_type)

        iterations_per_layer = args.iteration
        lim_0, lim_1 = args.jitter , args.jitter

        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps = 1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)

            aug_function = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ])
            inputs_jit = aug_function(inputs)

            # apply random jitter offsets
            off1 = random.randint(0, lim_0)
            off2 = random.randint(0, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()
            outputs = model_teacher(inputs_jit)

            # R_cross classification loss
            loss_ce = criterion(outputs, targets)

            # R_feature loss
            rescale = [args.first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
            loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

            # R_prior losses
            _, loss_var_l2 = get_image_prior_losses(inputs_jit)

            # l2 loss on images
            loss_l2 = torch.norm(inputs_jit.reshape(batch_size, -1), dim=1).mean()
            
            # combining losses
            loss_aux = args.tv_l2 * loss_var_l2 + \
                        args.l2_scale * loss_l2 + \
                        args.r_bn * loss_r_bn_feature

            loss = loss_ce + loss_aux

            if (iteration % save_every==0) or (iteration == iterations_per_layer-1):
                print("-------ipc_id {}---kk {}--iteration {}----------".format(ipc_id,kk,iteration))
                print("total loss", loss.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("main criterion", criterion(outputs, targets).item())
                # comment below line can speed up the training (no validation process)
                if hook_for_display is not None:
                    hook_for_display(inputs, targets)

            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            inputs.data = clip(inputs.data)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = inputs.data.clone() # using multicrop, save the last one
            best_inputs = denormalize(best_inputs)
            save_images(args, best_inputs, targets, ipc_id)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
    torch.cuda.empty_cache()

def save_images(args, images, targets, ipc_id):
    for id in range(images.shape[0]):
        if targets.ndimension() == 1:
            class_id = targets[id].item()
        else:
            class_id = targets[id].argmax().item()

        if not os.path.exists(args.syn_data_path):
            os.mkdir(args.syn_data_path)

        # save into separate folders
        dir_path = '{}/new{:03d}'.format(args.syn_data_path, class_id)
        place_to_store = dir_path +'/class{:03d}_id{:03d}.jpg'.format(class_id,ipc_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)

def validate(input, target, model):
    def accuracy(output, target, topk=(1,)):
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

    print("==> Verifier accuracy: ", prec1.item(),prec5.item())

def get_parser():

    parser = argparse.ArgumentParser(
        "SRe2L: recover data from pre-trained model")
    """Data save flags"""
    parser.add_argument('--exp-name', type=str, default='test',
                        help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument("--data", metavar="DIR", help="path to dataset")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
    parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
    parser.add_argument('--syn-data-path', type=str,
                        default='', help='where to store synthetic data')
    parser.add_argument('--store-best-images', action='store_true',
                        help='whether to store best images')
    """Optimization related flags"""
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--jitter', default=32, type=int, help='random shift on the synthetic data')
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
    parser.add_argument('--pretrained',type=str,default='',
                        help="PATH of the downloaded pretrained network")
    parser.add_argument('--start_epoch',type=int,default=0,
                        help="the start number of epoch")
    parser.add_argument('--index-start',type=int,default=0)
    parser.add_argument('--index-end',type=int,default=0)
    args = parser.parse_args()
    return args

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def validate_moco(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target= target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg

def load_state_dict(ckpt_path):
    if ckpt_path.startswith("https"):
        ckpt = torch.hub.load_state_dict_from_url(ckpt_path, progress=True)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return state_dict

def load_moco(arch, ckpt):
    model = torchvision.models.__dict__[arch]()
    state_dict = load_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=True)
    return model

def main_syn(args, ipc_id):
   
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    # create model
    print("=> creating model '{}'".format(args.arch_name))
    model = load_moco(args.arch_name, args.pretrained)

    model_teacher = model 
    model_teacher = model_teacher.to(my_device)
    model_teacher.eval() 
    for p in model_teacher.parameters():
        p.requires_grad = False 

    model_verifier = models.__dict__[args.verifier_arch](pretrained=True)
    model_verifier = model_verifier.cuda()
    model_verifier.eval()
    for p in model_verifier.parameters():
        p.requires_grad = False

    hook_for_display = lambda x,y: validate(x, y, model_verifier)
    get_images(args, model_teacher, hook_for_display, ipc_id)


if __name__ == '__main__':
    args = get_parser()
    args.syn_data_path= os.path.join(args.syn_data_path, args.exp_name)
    for ipc_id in range(args.index_start, args.index_end):
        print('==> main ipc = ', ipc_id)
        main_syn(args,ipc_id)

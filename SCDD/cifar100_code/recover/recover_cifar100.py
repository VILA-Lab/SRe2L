'''This code is modified from https://github.com/liuzechun/Data-Free-NAS'''

import os
import random
import argparse
import collections
import numpy as np
from PIL import Image

import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
import torch.utils.data.distributed
# import wandb

from utils import lr_cosine_policy, BNFeatureHook, clip_cifar, denormalize_cifar



def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    # if use_fp16:
    #     mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
    #     std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    # else:
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor



def get_images(args, model_teacher, hook_for_display, ipc_id):
    print("get_images call")
    save_every = 50
    batch_size = args.batch_size

    best_cost = 1e4


    loss_r_feature_layers = []
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module))

    # setup target labels
    # targets_all = torch.LongTensor(np.random.permutation(1000))
    targets_all = torch.LongTensor(np.arange(100))

    for kk in range(0, 100, batch_size):
        targets = targets_all[kk:min(kk+batch_size,100)].to('cuda')

        data_type = torch.float
        inputs = torch.randn((targets.shape[0], 3, 32, 32), requires_grad=True, device='cuda',
                             dtype=data_type)

        iterations_per_layer = args.iteration
        lim_0, lim_1 = args.jitter , args.jitter

        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps = 1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer) # 0 - do not use warmup
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)
            min_crop = 0.08
            max_crop = 1.0

            aug_function = transforms.Compose([
                # transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomResizedCrop(224, scale=(min_crop, max_crop)),
                # transforms.RandomHorizontalFlip(),
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
            # loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

            loss_r_bn_feature = [mod.r_feature.to(loss_ce.device) * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)]
            loss_r_bn_feature = torch.stack(loss_r_bn_feature).sum()

            # R_prior losses
            # _, loss_var_l2 = get_image_prior_losses(inputs_jit)

            # l2 loss on images
            # loss_l2 = torch.norm(inputs_jit.reshape(batch_size, -1), dim=1).mean()

            # combining losses
            # loss_aux = args.tv_l2 * loss_var_l2 + \
            #             args.l2_scale * loss_l2 + \
            #             args.r_bn * loss_r_bn_feature

            loss_aux = args.r_bn * loss_r_bn_feature

            loss = loss_ce + loss_aux

            if iteration % save_every==0 or iteration==(args.iteration-1):
                print("------------iteration {}----------".format(iteration))
                print("loss_ce", loss_ce.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("loss_total", loss.item())
                # comment below line can speed up the training (no validation process)
                if hook_for_display is not None:
                    acc_jit, _ = hook_for_display(inputs_jit, targets)
                    acc_image, loss_image = hook_for_display(inputs, targets)

                    metrics = {
                        'crop/acc_crop': acc_jit,
                        'image/acc_image': acc_image,
                        'image/loss_image': loss_image,
                    }
                    # wandb_metrics.update(metrics)

                metrics = {
                    'crop/loss_ce': loss_ce.item(),
                    'crop/loss_r_bn_feature': loss_r_bn_feature.item(),
                    'crop/loss_total': loss.item(),
                }
                # wandb_metrics.update(metrics)
                # wandb.log(wandb_metrics)


            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            inputs.data = clip_cifar(inputs.data)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = inputs.data.clone() # using multicrop, save the last one
            best_inputs = denormalize_cifar(best_inputs)
            save_images(args, best_inputs, targets, ipc_id)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        # exit()
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
    # print(f'input dim: {input.shape}')
    # return 0,0

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
        loss = nn.CrossEntropyLoss()(output, target)

    print("Verifier accuracy: ", prec1.item())
    return prec1.item(), loss.item()

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

def main_syn(args, ipc_id):
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)


    import torchvision
    model_teacher = torchvision.models.get_model(args.arch_name, num_classes=100)
    model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model_teacher.maxpool = nn.Identity()
    state_dict = load_state_dict(args.arch_path)
    model_teacher.load_state_dict(state_dict)

    print(model_teacher)
    # exit()

    # model_teacher = models.__dict__[args.arch_name](pretrained=True)
    # ckp = '/home/muxin/DD/cifar_train-main/save_ciai/resnet18_cifar100/ckpt.pth'
    # checkpoint = torch.load(ckp)
    # model_teacher.load_state_dict(checkpoint['state_dict'])

    # ckp = '/home/zeyuan/My-Dataset-Distillation/cifar_train/save/resnet18_E10/ckpt.pth'
    # checkpoint = torch.load(ckp)
    # model_teacher.load_state_dict(checkpoint['state_dict'])


    model_teacher = nn.DataParallel(model_teacher).cuda()

    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    if False:

        # model_verifier = MobileNetV2(num_classes=100)
        # model_verifier = nn.DataParallel(model_verifier).cuda()
        verifier_arch_path = '/home/muxin/DD/cifar_train-main/save_ciai/mobilenetv2_cifar100/ckpt.pth'
        # checkpoint = torch.load(ckp)
        # model_verifier.load_state_dict(checkpoint['state_dict'])

        model_verifier = models.__dict__['mobilenet_v2'](num_classes=100)
        model_verifier.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model_verifier.maxpool = nn.Identity()
        verifier_state_dict = load_state_dict(verifier_arch_path)
        model_verifier.load_state_dict(verifier_state_dict)


        model_verifier = model_verifier.cuda()

        # model_verifier = model_teacher #test
        model_verifier.eval()
        for p in model_verifier.parameters():
            p.requires_grad = False

        hook_for_display = lambda x,y: validate(x, y, model_verifier)
    else:
        hook_for_display = None
    get_images(args, model_teacher, hook_for_display, ipc_id)


def parse_args():
    parser = argparse.ArgumentParser(
        "SRe2L: recover data from pre-trained model")
    """Data save flags"""
    parser.add_argument('--exp-name', type=str, default='test',
                        help='name of the experiment, subfolder under syn_data_path')
    parser.add_argument('--syn-data-path', type=str,
                        default='/home/muxin/DD/SRe2L/recover/syn_data_new/cifar', help='where to store synthetic data')
    parser.add_argument('--store-best-images', action='store_true',
                        help='whether to store best images')
    """Optimization related flags"""
    parser.add_argument('--batch-size', type=int,
                        default=100, help='number of images to optimize at the same time')
    parser.add_argument('--iteration', type=int, default=1000,
                        help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for optimization')
    parser.add_argument('--jitter', default=4, type=int, help='random shift on the synthetic data')
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
    parser.add_argument('--arch-path', type=str, default='',
                        help='arch path from pretrained torchvision models')
    parser.add_argument('--verifier', action='store_true',
                        help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='mobilenet_v2',
                        help="arch name from torchvision models to act as a verifier")
    parser.add_argument('--GPU-ID', default='0', type=str)
    parser.add_argument('--ipc-start', default=0, type=int)
    parser.add_argument('--ipc-end', default=1, type=int)
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID

    args.syn_data_path= os.path.join(args.syn_data_path, args.exp_name)
    return args

if __name__ == '__main__':

    args = parse_args()

    # if not wandb.api.api_key:
    #     wandb.login(key='d15c9070e9dbb45d5decc4735216c27bb89f18a4')
    # wandb.init(project='sre2l-nips-re', name=args.exp_name)
    # global wandb_metrics
    # wandb_metrics = {}
    # for ipc_id in range(0,50):
    for ipc_id in range(args.ipc_start, args.ipc_end):
        print('ipc = ', ipc_id)
        # wandb.log({'ipc_id': ipc_id})
        main_syn(args, ipc_id)

    # wandb.finish()
import argparse
import collections
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import wandb
from PIL import Image
from torchvision import transforms

from utils import BNFeatureHook, lr_cosine_policy, tiny_clip, tiny_denormalize


def get_images(args, model_teacher, hook_for_display, ipc_id):
    print("get_images call")
    save_every = 100
    batch_size = args.batch_size

    best_cost = 1e4

    loss_r_feature_layers = []
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module))

    # setup target labels
    targets_all = torch.LongTensor(np.arange(200))

    for kk in range(0, 200, batch_size):
        targets = targets_all[kk : min(kk + batch_size, 200)].to("cuda")

        data_type = torch.float
        inputs = torch.randn((targets.shape[0], 3, 64, 64), requires_grad=True, device="cuda", dtype=data_type)

        iterations_per_layer = args.iteration
        lim_0, lim_1 = args.jitter, args.jitter

        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)  # 0 - do not use warmup
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)
            min_crop = 0.08
            max_crop = 1.0

            # strategy: start with whole image with mix crop of 1, then lower to 0.08
            # easy to hard
            min_crop = 0.08
            max_crop = 1.0
            if iteration < args.milestone * iterations_per_layer:
                if args.easy2hard_mode == "step":
                    min_crop = 1.0
                elif args.easy2hard_mode == "linear":
                    # min_crop linear decreasing: 1.0 -> 0.08
                    min_crop = 0.08 + (1.0 - 0.08) * (1 - iteration / (args.milestone * iterations_per_layer))
                elif args.easy2hard_mode == "cosine":
                    # min_crop cosine decreasing: 1.0 -> 0.08
                    min_crop = 0.08 + (1.0 - 0.08) * (1 + np.cos(np.pi * iteration / (args.milestone * iterations_per_layer))) / 2

            aug_function = transforms.Compose(
                [
                    # transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                    transforms.RandomResizedCrop(64, scale=(min_crop, max_crop)),
                    transforms.RandomHorizontalFlip(),
                ]
            )
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
            rescale = [args.first_bn_multiplier] + [1.0 for _ in range(len(loss_r_feature_layers) - 1)]
            loss_r_bn_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

            # combining losses
            loss_aux = args.r_bn * loss_r_bn_feature

            loss = loss_ce + loss_aux

            if (iteration % save_every == 0 or iteration == iterations_per_layer - 1) and hook_for_display is not None:
                print("------------iteration {}----------".format(iteration))
                print("loss_ce", loss_ce.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("loss_total", loss.item())
                if hook_for_display is not None:
                    acc_jit, _ = hook_for_display(inputs_jit, targets)
                    acc_image, loss_image = hook_for_display(inputs, targets)

                    metrics = {
                        "crop/acc_crop": acc_jit,
                        "image/acc_image": acc_image,
                        "image/loss_image": loss_image,
                    }
                    wandb_metrics.update(metrics)

                metrics = {
                    "crop/loss_ce": loss_ce.item(),
                    "crop/loss_r_bn_feature": loss_r_bn_feature.item(),
                    "crop/loss_total": loss.item(),
                }
                wandb_metrics.update(metrics)
                wandb.log(wandb_metrics)

            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            inputs.data = tiny_clip(inputs.data)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = inputs.data.clone()  # using multicrop, save the last one
            best_inputs = tiny_denormalize(best_inputs)
            save_images(args, best_inputs, targets, ipc_id)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)
        if args.verifier:
            exit()
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
        dir_path = "{}/new{:03d}".format(args.syn_data_path, class_id)
        place_to_store = dir_path + "/class{:03d}_id{:03d}.jpg".format(class_id, ipc_id)
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
        loss = nn.CrossEntropyLoss()(output, target)

    print("Verifier accuracy: ", prec1.item())
    return prec1.item(), loss.item()


def main_syn(args, ipc_id):
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    import torchvision

    model = torchvision.models.get_model(args.arch_name, weights=False, num_classes=200)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()

    checkpoint = torch.load(args.arch_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    model_teacher = nn.DataParallel(model).cuda()
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    if args.verifier:
        model = models.__dict__["mobilenet_v2"](pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 200)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        checkpoint = torch.load("/path/to/mobilenetv2_tiny/checkpoint.pth", map_location="cpu")

        model.load_state_dict(checkpoint["model"])

        model_verifier = model.cuda()
        model_verifier.eval()
        for p in model_verifier.parameters():
            p.requires_grad = False

        hook_for_display = lambda x, y: validate(x, y, model_verifier)
    else:
        hook_for_display = None
    get_images(args, model_teacher, hook_for_display, ipc_id)


def parse_args():
    parser = argparse.ArgumentParser("CDA for ImageNet-1K")
    """Data save flags"""
    parser.add_argument("--exp-name", type=str, default="test", help="name of the experiment, subfolder under syn_data_path")
    parser.add_argument("--syn-data-path", type=str, default="./syn-data", help="where to store synthetic data")
    parser.add_argument("--store-best-images", action="store_true", help="whether to store best images")
    """Optimization related flags"""
    parser.add_argument("--batch-size", type=int, default=100, help="number of images to optimize at the same time")
    parser.add_argument("--iteration", type=int, default=1000, help="num of iterations to optimize the synthetic data")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate for optimization")
    parser.add_argument("--jitter", default=4, type=int, help="random shift on the synthetic data")
    parser.add_argument("--r-bn", type=float, default=0.05, help="coefficient for BN feature distribution regularization")
    parser.add_argument("--first-bn-multiplier", type=float, default=10.0, help="additional multiplier on first bn layer of R_bn")
    """Model related flags"""
    parser.add_argument("--arch-name", type=str, default="resnet18", help="arch name from pretrained torchvision models")
    parser.add_argument("--arch-path", type=str, default="")
    parser.add_argument("--verifier", action="store_true", help="whether to evaluate synthetic data with another model")
    parser.add_argument("--verifier-arch", type=str, default="mobilenet_v2", help="arch name from torchvision models to act as a verifier")
    parser.add_argument("--easy2hard-mode", default="cosine", type=str, choices=["step", "linear", "cosine"])
    parser.add_argument("--milestone", default=0, type=float)
    parser.add_argument("--G", default="-1", type=str)
    parser.add_argument("--ipc-start", default=0, type=int)
    parser.add_argument("--ipc-end", default=1, type=int)
    parser.add_argument("--wandb-key", default="", type=str)
    args = parser.parse_args()

    assert args.milestone >= 0 and args.milestone <= 1

    if args.G != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.G
        print("set CUDA_VISIBLE_DEVICES to ", args.G)

    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    return args


if __name__ == "__main__":
    args = parse_args()

    if not wandb.api.api_key:
        wandb.login(key=args.wandb_key)
    wandb.init(project="cda-gen-tiny", name=args.exp_name)
    global wandb_metrics
    wandb_metrics = {}
    # for ipc_id in range(0,50):
    for ipc_id in range(args.ipc_start, args.ipc_end):
        print("ipc = ", ipc_id)
        wandb.log({"ipc_id": ipc_id})
        main_syn(args, ipc_id)

    wandb.finish()
    print("Done.")

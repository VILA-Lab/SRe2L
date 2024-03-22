'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR

import os
import argparse

from utils import progress_bar

import numpy as np
import math 


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--output-dir', default='./save', type=str)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--check-ckpt', default=None, type=str)

parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--train-dir',default='',type=str)
parser.add_argument('--teacher-model',default='resnet18',type=str)
parser.add_argument('--teacher-model-dir',default='',type=str)
parser.add_argument('--student-model', type=str,default='resnet18', help='student model name')
parser.add_argument('--re-epochs', default=300, type=int)
args = parser.parse_args()

if args.check_ckpt:
    checkpoint = torch.load(args.check_ckpt)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(f'==> test ckp: {args.check_ckpt}, acc: {best_acc}, epoch: {start_epoch}')
    exit()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

from torchvision.datasets import ImageFolder
from imagenet_ipc import ImageFolderIPC
trainset = ImageFolderIPC(root=args.train_dir, transform=transform_train, image_number=50)

print(trainset.__len__())
print(trainset.class_to_idx)


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2) # origin bs 128

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2) # origin bs 100

# Model
print('==> Building model..')

model = torchvision.models.get_model(args.student_model, num_classes=100)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Identity()

net = model.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


import torchvision
model_teacher = torchvision.models.get_model(args.teacher_model, num_classes=100)
model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model_teacher.maxpool = nn.Identity()

model_teacher = nn.DataParallel(model_teacher).cuda()

ckp = args.teacher_model_dir 
checkpoint = torch.load(ckp)
model_teacher.load_state_dict(checkpoint['state_dict'])


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)

if True:
    scheduler = LambdaLR(
        optimizer,
        lambda step: 0.5 * (1.0 + math.cos(math.pi * step / args.re_epochs / 2))
        if step <= args.re_epochs
        else 0,
        last_epoch=-1,
    )
else:
    scheduler = LambdaLR(
        optimizer,
        lambda step: (1.0 - step / args.re_epochs) if step <= args.re_epochs else 0,
        last_epoch=-1,
    )


def mixup_data(x, y, alpha=0.8):
    """
    Returns mixed inputs, mixed targets, and mixing coefficients.
    For normal learning
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, cutmix_prob=1, beta=1.0):
    r = np.random.rand(1)
    if beta > 0 and r < cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(x.size()[0]).cuda()
        # target_a = target
        # target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    return x

args.temperature = 30
loss_function_kl = nn.KLDivLoss(reduction='batchmean')
# Training
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs, target_a, target_b, lam = mixup_data(inputs, targets)

        optimizer.zero_grad()
        outputs = net(inputs)

        soft_label = model_teacher(inputs).detach()
        outputs_ = F.log_softmax(outputs / args.temperature, dim=1)
        soft_label = F.softmax(soft_label / args.temperature, dim=1)

        loss = loss_function_kl(outputs_, soft_label)


        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100.*correct/total
            if acc > best_acc:
                best_acc = acc

    # Save checkpoint.
    acc = 100.*correct/total
    print('epoch %d | Acc: %.3f' % (epoch,acc))

    if acc > best_acc:
    # if True:
        print('Saving..')
        state = {
            'state_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }


        path = os.path.join(args.output_dir, './ckpt.pth')
        torch.save(state, path)
        best_acc = acc


import time
start_time = time.time()
for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    if epoch % 10 == 0 or epoch == args.epochs - 1:
        test(epoch)
    scheduler.step()
end_time = time.time()
print(f'total time: {end_time - start_time}')
print('Best Acc: %.3f' % (best_acc))
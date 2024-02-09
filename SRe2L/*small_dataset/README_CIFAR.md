# CIFAR-100 Experiments

## Squeeze

For the squeezed model on CIFAR-100, we modify the ResNet model structure and adapt CIFAR-10 training code available at <https://github.com/kuangliu/pytorch-cifar> to train the mode on CIFAR-100.

```python
model = torchvision.models.get_model("resnet18", num_classes=100)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Identity()
```

```bash
sh squeeze_cifar.sh
```

|   name    | epochs | acc@1 (last) |
| :-------: | :----: | :----------: |
| ResNet-18 |  200   |    78.72     |

## Recover

To recover the CIFAR, we adapt the recovery code on ImageNet-1K and tune the hyperparameters for CIFAR. The difference is that we discard the Multi-crop Optimization and optimize the whole image due to the small resolution of CIFAR.

```bash
sh recover_cifar.sh
```

## Relabel

Based on the squeezed code, we add corresponding relabeling code to relabel the CIFAR-100 data.

```bash
# modify the --teacher-path (obtained from the squeeze phase) and --syn-data-path (obtained from the recover phase) in relabel_cifar.sh
sh relabel_cifar.sh
```

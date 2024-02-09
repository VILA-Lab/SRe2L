# Tiny-ImageNet Experiments

## Squeeze

For the squeezed model on Tiny-ImageNet, we modified the ResNet model structure and adapted the official torchvision classification code available at <https://github.com/pytorch/vision/tree/main/references/classification> to train the model on Tiny-ImageNet.

```python
model = models.__dict__['resnet18'](num_classes=200)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Identity()
```

You can find the training code and checkpoints at [tiny-imagenet repo](https://github.com/zeyuanyin/tiny-imagenet).

```bash
git clone git@github.com:zeyuanyin/tiny-imagenet.git
cd tiny-imagenet/script
sh train.sh
```

## Recover

To recover the Tiny-ImageNet data, we adapted the recovery code on ImageNet-1K and tuned the hyperparameters for Tiny-ImageNet.

```bash
sh recover_tiny.sh
```

## Relabel

Based on the squeezed code, we added corresponding relabeling code to relabel the Tiny-ImageNet data, available at [tiny-imagenet/classification/train_kd.py](https://github.com/zeyuanyin/tiny-imagenet/blob/main/classification/train_kd.py)

```bash
git clone git@github.com:zeyuanyin/tiny-imagenet.git
cd tiny-imagenet/script
# modify the --teacher-path (obtained from the squeeze phase) and --syn-data-path (obtained from the recover phase) in train_kd.sh
sh train_kd.sh
```

The validation (post-training) logs are available at [tiny-imagenet/logs](https://github.com/zeyuanyin/tiny-imagenet/tree/main/log).

## Download distilled images from [![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/zeyuanyin/SRe2L)

|      dataset      | resolution | iteration |  IPC  |                                                files                                                 |
| :---------------: | :--------: | :-------: | :---: | :--------------------------------------------------------------------------------------------------: |
| Tiny-ImageNet-200 |   64x64    |    1K     |  50   | [images](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/sre2l_tiny_rn18_1k_ipc50.zip)  |
| Tiny-ImageNet-200 |   64x64    |    4K     |  100  | [images](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/sre2l_tiny_rn18_4k_ipc100.zip) |

# SRe2L

Official PyTorch implementation of paper (NeurIPS 2023 spotlight):
>[__"*Squeeze*, *Recover* and *Relabel*: Dataset Condensation at ImageNet Scale From A New Perspective"__](https://arxiv.org/abs/2306.13092)<br>
>[Zeyuan Yin](https://zeyuanyin.github.io), [Eric Xing](http://www.cs.cmu.edu/~epxing/), [Zhiqiang Shen](http://zhiqiangshen.com/)<br>
>MBZUAI, CMU

[`[Project Page]`](https://zeyuanyin.github.io/projects/SRe2L/)  [`[Paper]`](https://arxiv.org/abs/2306.13092)

<div align=center>
<img width=80% src="./img/overview.png"/>
</div>


## Catalog
- [x] ImageNet-1K Code
- [x] Tiny-ImageNet-200 Code
- [x] CIFAR-100 Code
- [x] FKD-Mix Code
- [x] Naive KD Code
- [x] Distilled Datasets [![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/zeyuanyin/SRe2L)

## Abstract

We present a new dataset condensation framework termed *<b>S</b>queeze* (<img width=2% src="./img/squeeze.png"/>), *<b>Re</b>cover* (<img width=2% src="./img/recover.png"/>) and *<b>Re</b>labe<b>l</b>* (<img width=1.7% src="./img/relabel.png"/>) (SRe<sup>2</sup>L) that decouples the bilevel optimization of model and synthetic data during training, to handle varying scales of datasets, model architectures and image resolutions for effective dataset condensation. The proposed method demonstrates flexibility across diverse dataset scales and exhibits multiple advantages in terms of arbitrary resolutions of synthesized images, low training cost and memory consumption with high-resolution training, and the ability to scale up to arbitrary evaluation network architectures. Extensive experiments are conducted on Tiny-ImageNet and full ImageNet-1K datasets. Under 50 IPC, our approach achieves the highest 42.5% and 60.8% validation accuracy on Tiny-ImageNet and ImageNet-1K, outperforming all previous state-of-the-art methods by margins of 14.5% and 32.9%, respectively. Our approach also outperforms MTT by approximately 52&times; (ConvNet-4) and 16&times; (ResNet-18) faster in speed with less memory consumption of 11.6&times; and 6.4&times; during data synthesis.


## Distillation Animation

<div align=left>
<img style="width:70%" src="https://github.com/zeyuanyin/Public-Large-Files/releases/download/SRe2L/syn_img_gif.gif">
</div>

******************************
Kindly wait a few seconds for the animation visualizations to load.
******************************

## Distilled ImageNet

<div align=left>
<img style="width:70%" src="./img/animation.gif">
</div>

## Squeeze <img width=2.8% src="./img/squeeze.png"/>

- For ImageNet-1K, we use the official PyTorch pre-trained models from [Torchvision Model Zoo](https://pytorch.org/vision/stable/models.html).

- For Tiny-ImageNet-200, we adapt official [Torchvision code](https://github.com/pytorch/vision/tree/main/references/classification) to train the model from scratch. You can find the training code and checkpoints at [tiny-imagenet](https://github.com/zeyuanyin/tiny-imagenet).


## Recover <img width=2.8% src="./img/recover.png"/>

More details in [recover/README.md](recover/README.md).
```bash
cd recover
sh recover.sh
```

## Relabel <img width=2.3% src="./img/relabel.png"/>

More details in [relabel/README.md](relabel/README.md).
```bash
cd relabel
sh relabel.sh
```

## Validate distilled dataset

We provide two kinds of validation code: FKD and Naive KD. FKD is the main validation code aligned with our paper for relabeled distilled images. Naive KD is an alternative validation code to quickly validate the performance of the distilled data without the relabel process.
More details in [validate/README.md](validate/README.md).
```bash
cd validate
sh train_FKD.sh
```

## Download

You can download distilled data and soft labels from [![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/zeyuanyin/SRe2L).

| dataset | resolution | iteration | IPC | files |
|:---:|:---:|:---:|:---:| :---:|
| ImageNet-1K | 224x224 | 4K | 50 | [images](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/sre2l_in1k_rn18_4k_ipc50.zip) <br> [mixup labels](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/FKD_mixup_fp16.tar.gz) / [cutmix labels](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/FKD_cutmix_fp16.tar.gz) |
| ImageNet-1K | 224x224 | 2K | 50 | [images](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/sre2l_in1k_rn18_2k_ipc50.zip)|
| ImageNet-1K | 224x224 | 4K | 200 | [images](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/sre2l_in1k_rn18_4k_ipc200.zip)|
| Tiny-ImageNet-200 | 64x64 | 1K | 50 | [images](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/sre2l_tiny_rn18_1k_ipc50.zip)|
| Tiny-ImageNet-200 | 64x64 | 4K | 100 | [images](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/sre2l_tiny_rn18_4k_ipc100.zip)|


## Results

Our Top-1 accuracy (%) under different IPC settings on Tiny-ImageNet and ImageNet-1K datasets:

<div align=center>
<img style="width:50%" src="./img/results.png">
</div>


## Citation

If you find our code useful for your research, please cite our paper.

```
@inproceedings{yin2023squeeze,
  title={Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective},
  author={Yin, Zeyuan and Xing, Eric and Shen, Zhiqiang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```


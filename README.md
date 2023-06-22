# SRe2L: *<b>S</b>queeze* <img width=3.1% src="./img/squeeze.png"/> - *<b>Re</b>cover* <img width=3.1% src="./img/recover.png"/> - *<b>Re</b>labe<b>l</b>* <img width=2.5% src="./img/relabel.png"/>

Official PyTorch implementation of paper [__"*Squeeze*, *Recover* and *Relabel*: Dataset Condensation at ImageNet Scale From A New Perspective"__](), [Zeyuan Yin](https://zeyuanyin.github.io),     [Eric Xing](http://www.cs.cmu.edu/~epxing/), and     [Zhiqiang Shen](http://zhiqiangshen.com/).

[`[Project Page]`](https://zeyuanyin.github.io/projects/SRe2L/)  [`[Paper]`]()

<div align=center>
<img width=80% src="./img/overview.png"/>
</div>

## Abstract

We present a new dataset condensation framework termed *<b>S</b>queeze* (<img width=2% src="./img/squeeze.png"/>), *<b>Re</b>cover* (<img width=2% src="./img/recover.png"/>) and *<b>Re</b>labe<b>l</b>* (<img width=1.7% src="./img/relabel.png"/>) (SRe<sup>2</sup>L) that decouples the bilevel optimization of model and synthetic data during training, to handle varying scales of datasets, model architectures and image resolutions for effective dataset condensation. The proposed method demonstrates flexibility across diverse dataset scales and exhibits multiple advantages in terms of arbitrary resolutions of synthesized images, low training cost and memory consumption with high-resolution training, and the ability to scale up to arbitrary evaluation network architectures. Extensive experiments are conducted on Tiny-ImageNet and full ImageNet-1K datasets. Under 50 IPC, our approach achieves the highest 42.5% and 60.8% validation accuracy on Tiny-ImageNet and ImageNet-1K, outperforming all previous state-of-the-art methods by margins of 14.5% and 32.9%, respectively. Our approach also outperforms MTT by approximately 52&times; (ConvNet-4) and 16&times; (ResNet-18) faster in speed with less memory consumption of 11.6&times; and 6.4&times; during data synthesis.


## Squeeze <img width=2.8% src="./img/squeeze.png"/>

- For ImageNet-1K, we use the official PyTorch pre-trained models from [Torchvision Model Zoo](https://pytorch.org/vision/stable/models.html).

- For Tiny-ImageNet-200, we use official [Torchvision code](https://github.com/pytorch/vision/tree/main/references/classification) to train the model from scratch.


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

## Train on Distilled Data

More details in [train/README.md](train/README.md).
```bash
cd train
sh train.sh
```

## Download

You can download distilled data and soft labels from https://zeyuanyin.github.io/projects/SRe2L/#Download


## Citation

```
@article{yin2023squeeze,
	title = {Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective},
	author = {Yin, Zeyuan and Xing, Eric and Shen, Zhiqiang},
	journal = {arXiv preprint},
	year = {2023}
}
```


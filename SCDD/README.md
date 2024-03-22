# Self-supervised Compression Method for Dataset Distillation 

This repository hosts the code for our research paper titled *A Good Compression Is All You Need for Dataset Distillation* on ImageNet-1k and CIFAR-100 datasets.

<div align=center>
<img width=80% src="./source/vis_all.jpg"/>
</div>


## Abstract
Dataset distillation aims to *compress* information and knowledge from a large-scale original dataset to a new compact dataset while striving to preserve the utmost degree of the original data's informational essence. Previous studies have predominantly concentrated on aligning the intermediate statistics between the original and distilled data, such as weight trajectory, features, gradient, BatchNorm, etc. 

In this work, we consider addressing this task through the new lens of **model informativeness** in compression on the original dataset pretraining. We observe that with the prior state-of-the-art SRe&sup2;L, as model sizes increase, it becomes increasingly challenging for supervised pretrained models to retrieve learned information during data synthesis, as the channel-wise mean and variance inside the model are flatting and less informative. Building on this observation, we introduce SC-DD, a **S**elf-supervised **C**ompression method for **D**ataset **D**istillation that facilitates diverse information compression and recovery compared to traditional supervised learning schemes, further reaps the potential of large pretrained models with enhanced capabilities. 

Extensive experiments are conducted on CIFAR-100, Tiny-ImageNet and ImageNet-1K datasets to demonstrate the superiority of our proposed approach. The proposed SC-DD outperforms all previous state-of-the-art supervised dataset distillation methods when employing larger models, such as SRe&sup2;L, MTT, TESLA, DC, CAFE, etc., by large margins. 

<div align=center>
<img width=80% src="./source/method.jpg"/>
</div>

Overview of our learning paradigm. The top-left subfigure is the paradigm of supervised pertaining with an end-to-end training scheme for both the backbone network and final alignment classifier. The bottom-left subfigure is the paradigm of our proposed procedure for dataset distillation: a backbone model is first pretrained using a self-supervised objective, then a linear probing layer is adjusted to align the distribution of pertaining and target dataset distribution. We do not fine-tune the backbone during the alignment phase to preserve the better intermediate distributions of mean and variance in batch normalization layers (illustrated in the middle yellow line chart of the figure). The bottom-middle subfigure is the data synthesis procedure and the left subfigure is the visualization of distilled images.

## Synthetic Images

Below, we present our synthetic images generated for ImageNet-1k, which subjectively closely resemble real images.

<div align=center>
<img style="width:80%" src="./source/syn_show.gif">
</div>

## Experiments
### Pretrain Model
The command below executes the training of the MoCo v2 model on the ImageNet-1k dataset.
```bash
python main_moco.py -a resnet18 --lr 0.03 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 /your/path/imagenet --mlp --moco-t 0.2 --aug-plus --cos 
```

### Recover
The subsequent command will execute the recovery code on ImageNet-1k to synthesize images.
```bash
python data_synthesis.py --arch-name "resnet50" --exp-name "recover_resnet50_ipc50" --pretrained "/your/PretrainedModel.pth.tar" --syn-data-path './syn_data' --first-bn-multiplier 10 --batch-size 50 --lr 0.1 --iteration 1000 --l2-scale 0 --tv-l2 0 --r-bn 0.01 --verifier --store-best-images --index-start 0 --index-end 50 
```

The subsequent command will execute the recovery code on CIFAR-100.
```bash
python recover_cifar100.py --arch-name "resnet18" --arch-path "/your/path/model_ckpt.pth" --exp-name "cifar100_ipc50" --batch-size 100 --lr 0.4 --iteration 1000 --l2-scale 0 --tv-l2 0 --r-bn 0.005 --store-best-images --ipc-start 0 --ipc-end 1 --GPU-ID 0  
```

### Validation
The following command executes the code for post-training on the ImageNet-1k dataset.
```bash
python train_kd.py --batch-size 64 --model resnet18 --teacher-model resnet18 --epochs 1000 --cos -j 8 --gradient-accumulation-steps 1 -T 20 --mix-type 'cutmix' --val-dir /your/path/imagenet/val --train-dir /your/synthesis_data_path --output-dir ./save --image-select-idx 0 
```

The following command executes the code for post-training on the CIFAR-100 dataset.
```bash
python post_cifar100.py --epochs 200 --lr 0.005 --student-model resnet18 --teacher-model resnet18 --teacher-model-dir '/your/path/resnet_18_ckpt.pth' --train-dir '/your/path/syn_data' --output-dir /your/path/save 
```

## Performance
Here is the table displaying the Top-1 validation accuracy obtained from training with 10, 50, 100, and 200 synthetic images per class using ResNet18.

| IPC | CIFAR-100 | Tiny-ImageNet | ImageNet-1K |
|---|---|---|---|
| 10 img/cls | - | 31.6 | 32.1 |
| 50 img/cls | 53.4 | 45.9 | 53.1 |
| 100 img/cls | - | - | 57.9 |
| 200 img/cls | - | - | 63.5 | 


## Acknowledgments
Our code framework is derived from [https://github.com/VILA-Lab/SRe2L/tree/main/SRe2L](https://github.com/VILA-Lab/SRe2L/tree/main/SRe2L).

The code in the `/SCDD/imagenet_code/pretrain_moco` section is adapted from [https://github.com/facebookresearch/moco](https://github.com/facebookresearch/moco).
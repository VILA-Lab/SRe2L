# Recover knowledge & synthesize data from Pre-trained models

## Preparation

- Python >= 3.8
- PyTorch >= 2.0.0
- Torchvision >= 0.15.1


## Distilled Data Generation

To recover knowledge from a pre-trained model and synthersize distilled data, run `recover.sh` with the desired arguments.

```bash
python data_synthesis.py \
    --arch-name "resnet18" \
    --exp-name "rn18_bn0.01_[4K]_x_l2_x_tv.crop" \
    --batch-size 100 \
    --lr 0.25 \
    --iteration 4000 \
    --l2-scale 0 --tv-l2 0 --r-bn 0.01 \
    --verifier --store-best-images
```

## Usage

```
usage: data_synthesis.py

[-h] [--exp-name EXP_NAME] [--syn-data-path SYN_DATA_PATH] [--store-best-images] [--batch-size BATCH_SIZE] [--iteration ITERATION] [--lr LR] [--jitter JITTER] [--r-bn R_BN] [--first-bn-multiplier FIRST_BN_MULTIPLIER][--tv-l2 TV_L2] [--l2-scale L2_SCALE] [--arch-name ARCH_NAME] [--verifier] [--verifier-arch VERIFIER_ARCH]

arguments:
  -h, --help            show this help message and exit
  --exp-name EXP_NAME   name of the experiment, subfolder under syn_data_path
  --syn-data-path SYN_DATA_PATH
                        where to store synthetic data
  --store-best-images   whether to store best images
  --batch-size BATCH_SIZE
                        number of images to optimize at the same time
  --iteration ITERATION
                        num of iterations to optimize the synthetic data
  --lr LR               learning rate for optimization
  --jitter JITTER       random shift on the synthetic data
  --r-bn R_BN           coefficient for BN feature distribution regularization
  --first-bn-multiplier FIRST_BN_MULTIPLIER
                        additional multiplier on first bn layer of R_feature
  --tv-l2 TV_L2         coefficient for total variation L2 loss
  --l2-scale L2_SCALE   l2 loss on the image
  --arch-name ARCH_NAME
                        arch name from pretrained torchvision models
  --verifier            whether to evaluate synthetic data with another model
  --verifier-arch VERIFIER_ARCH
                        arch name from torchvision models to act as a verifier
```



## Download Distilled Data from https://zeyuanyin.github.io/projects/SRe2L/#Download

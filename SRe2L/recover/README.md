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

## Multi-GPU training (New Setting)
**[Suggestion for [multi-GPU training](https://github.com/VILA-Lab/SRe2L/issues/1#issuecomment-1649766741)]:**
[Xiaochen Ma](https://github.com/SunnyHaze) has implemented a set of Multi-GPU training code for the recovery stage with `DistributedSampler` for SRe2L, without using `DDP` or `DP`. It allows for easy and efficient multi-GPU inference with custom `IPC (Image Per Class)` parameters. 

Please switch the working directory to `./multi_gpu` and run the `multi_gput_recover.sh` script in that directory to start the multi-GPU training.

```bash
base_dir="./recover_imagenet1k_resnet18_ipc50_iter4000"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
data_synthesis.py \
    --arch-name "resnet18" \
    --exp-name ${base_dir} \
    --batch-size 100 \
    --lr 0.25 \
    --iteration 4000 \
    --ipc 50 \
    --l2-scale 0 --tv-l2 0 --r-bn 0.01 \
    --verifier --store-best-images \
2> ${base_dir}_error.log 1>${base_dir}_logs.log
```

> [!NOTE]\
> You have to modify the CUDA related parameter to fit your own device.

Here are help list for parameters:

```bash
usage: SRe2L: recover data from pre-trained model [-h] [--exp-name EXP_NAME]
                                                  [--syn-data-path SYN_DATA_PATH]
                                                  [--print_period PRINT_PERIOD]
                                                  [--store-best-images]
                                                  [--batch-size BATCH_SIZE]
                                                  [--iteration ITERATION]
                                                  [--lr LR]
                                                  [--image_shifting IMAGE_SHIFTING]
                                                  [--r-bn R_BN]
                                                  [--first-bn-multiplier FIRST_BN_MULTIPLIER]
                                                  [--tv-l2 TV_L2]
                                                  [--l2-scale L2_SCALE]
                                                  [--arch-name ARCH_NAME]
                                                  [--verifier]
                                                  [--verifier-arch VERIFIER_ARCH]
                                                  [--ipc IPC]
                                                  [--num-classes NUM_CLASSES]
                                                  [--world-size WORLD_SIZE]
                                                  [--dist-url DIST_URL]
                                                  [--dist-backend DIST_BACKEND]

optional arguments:
  -h, --help            show this help message and exit
  --exp-name EXP_NAME   name of the experiment, subfolder under syn_data_path
  --syn-data-path SYN_DATA_PATH
                        where to store synthetic data
  --print_period PRINT_PERIOD
                        print period, count by iteration
  --store-best-images   whether to store best images
  --batch-size BATCH_SIZE
                        number of images to optimize at the same time
  --iteration ITERATION
                        num of iterations to optimize the synthetic data
  --lr LR               learning rate for optimization
  --image_shifting IMAGE_SHIFTING
                        random shift on the synthetic data
  --r-bn R_BN           coefficient for BN feature distribution regularization
  --first-bn-multiplier FIRST_BN_MULTIPLIER
                        additional multiplier on first bn layer of R_bn
  --tv-l2 TV_L2         coefficient for total variation L2 loss
  --l2-scale L2_SCALE   l2 loss on the image
  --arch-name ARCH_NAME
                        arch name from pretrained torchvision models
  --verifier            whether to evaluate synthetic data with another model
  --verifier-arch VERIFIER_ARCH
                        arch name from torchvision models to act as a verifier
  --ipc IPC             IPC, Image Per Class
  --num-classes NUM_CLASSES
                        number of classes in the dataset
  --world-size WORLD_SIZE
                        number of distributed processes/gpus
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
```

## Multi-GPU training (Previous Setting)
We provide but do not recommend using DataParallel across multiple GPUs due to the delays incurred by parallelization.
Instead, we also provide an `IPC (Image Per Class)` control setting and suggest using a single GPU to synthesize images under a specific `IPC` range.
For instance, to synthesize a dataset of 100 IPC on 4 GPUs, you can maximize the utilization of each GPU by running the programs on each with a distinct `IPC` range, via executing the following commands separately on each GPU:

```bash
# GPU-0
CUDA_VISIBLE_DEVICES=0 \
python data_synthesis.py --ipc-start 0 --ipc-end 25 \
  --arch-name "resnet18" \
  ...

# GPU-1
CUDA_VISIBLE_DEVICES=1 \
python data_synthesis.py --ipc-start 25 --ipc-end 50 \
  --arch-name "resnet18" \
  ...

# GPU-2
CUDA_VISIBLE_DEVICES=2 \
python data_synthesis.py --ipc-start 50 --ipc-end 75 \
  --arch-name "resnet18" \
  ...

# GPU-3
CUDA_VISIBLE_DEVICES=3 \
python data_synthesis.py --ipc-start 75 --ipc-end 100 \
  --arch-name "resnet18" \
  ...
```

```
usage: data_synthesis.py

[-h] [--exp-name EXP_NAME] [--syn-data-path SYN_DATA_PATH] [--store-best-images] [--batch-size BATCH_SIZE] [--iteration ITERATION] [--lr LR] [--jitter JITTER] [--r-bn R_BN] [--first-bn-multiplier FIRST_BN_MULTIPLIER][--tv-l2 TV_L2] [--l2-scale L2_SCALE] [--arch-name ARCH_NAME] [--verifier] [--verifier-arch VERIFIER_ARCH] [--ipc-start IPC_START] [--ipc-end IPC_END]

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
  --ipc-start IPC_START
                        start index of IPC
  --ipc-end IPC_END     end index of IPC
```



## Download distilled images from [![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/zeyuanyin/SRe2L)


| dataset | resolution | iteration | IPC | files |
|:---:|:---:|:---:|:---:| :---:|
| ImageNet-1K | 224x224 | 4K | 50 | [images](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/sre2l_in1k_rn18_4k_ipc50.zip)|
| ImageNet-1K | 224x224 | 2K | 50 | [images](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/sre2l_in1k_rn18_2k_ipc50.zip)|
| ImageNet-1K | 224x224 | 4K | 200 | [images](https://huggingface.co/datasets/zeyuanyin/SRe2L/resolve/main/sre2l_in1k_rn18_4k_ipc200.zip)|

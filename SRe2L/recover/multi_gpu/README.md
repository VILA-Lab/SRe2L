## Multi-GPU training (New Setting)
**[Suggestion for [multi-GPU training](https://github.com/VILA-Lab/SRe2L/issues/1#issuecomment-1649766741)]:**
[Xiaochen](https://github.com/SunnyHaze) has implemented a set of Multi-GPU training code for the recovery stage with `DistributedSampler` for SRe2L, without using `DDP` or `DP`. It allows for easy and efficient multi-GPU inference with custom `IPC (Image Per Class)` parameters. 

To start multi-GPU recover training, please run the `multi_gput_recover.sh` script in that directory.

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

# wandb disabled
wandb enabled
wandb online

python train_FKD.py \
    --wandb-project 'final_rn18_fkd' \
    --batch-size 1024 \
    --model resnet18 \
    --cos \
    -j 4 --gradient-accumulation-steps 2 \
    -T 20 \
    --mix-type 'cutmix' \
    --output-dir ./save/final_rn18_fkd/rn18_[4K]_T20/ \
    --train-dir ../recover/syn_data/rn18_bn0.01_[4K]_x_l2_x_tv.crop \
    --val-dir /path/to/imagenet/val \
    --fkd-path ../relabel/FKD_cutmix_fp16
# wandb disabled
wandb enabled
wandb online

python train_KD.py \
    --wandb-project 'val_rn18_kd' \
    --batch-size 512 \
    --gradient-accumulation-steps 2 \
    --model resnet18 \
    --teacher-model resnet18 \
    --cos \
    -j 4 \
    -T 20 \
    --mix-type 'cutmix' \
    --output-dir ./save/val_rn18_kd/rn18_[4K]_T20 \
    --train-dir ../recover/syn_data/rn18_bn0.01_[4K]_x_l2_x_tv.crop \
    --val-dir /home/zeyuan/imagenet/val

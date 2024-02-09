python relabel_cifar.py \
    --epochs 400 \
    --output-dir ./save_post_cifar100/ipc50 \
    --syn-data-path /path/to/syn_data/cifar100_rn18_1K_mobile.lr0.25.bn0.01 \
    --teacher-path /path/to/cifar100/resnet18_E200/ckpt.pth' \
    --ipc 50 --batch-size 128

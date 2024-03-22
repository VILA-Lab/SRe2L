python recover_cifar100.py \
--arch-name "resnet18" \
--arch-path "/your/path/model_ckpt.pth" \
--exp-name "cifar100_ipc50" \
--batch-size 100 \
--lr 0.4 \
--iteration 1000 \
--l2-scale 0 --tv-l2 0 --r-bn 0.005 \
--store-best-images \
--ipc-start 0 --ipc-end 1 --GPU-ID 0 
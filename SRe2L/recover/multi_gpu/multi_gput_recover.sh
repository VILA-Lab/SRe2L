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
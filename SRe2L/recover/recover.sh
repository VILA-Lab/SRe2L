python data_synthesis.py \
    --arch-name "resnet18" \
    --exp-name "rn18_bn0.01_[4K]_x_l2_x_tv.crop" \
    --batch-size 100 \
    --lr 0.25 \
    --iteration 4000 \
    --l2-scale 0 --tv-l2 0 --r-bn 0.01 \
    --verifier --store-best-images

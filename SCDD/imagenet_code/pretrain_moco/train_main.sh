CUDA_VISIBLE_DEVICES=0,1 \
python main_moco.py \
  -a resnet18 \
  --lr 0.03 \
  --batch-size 16 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /data/ironman/zeyuan/imagenet \
  --mlp --moco-t 0.2 --aug-plus --cos 
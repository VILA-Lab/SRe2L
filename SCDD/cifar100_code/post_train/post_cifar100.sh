python post_cifar100.py \
--epochs 200 \
--lr 0.005 \
--student-model resnet18 \
--teacher-model resnet18 \
--teacher-model-dir 'your/path/resnet_18_ckpt.pth' \
--train-dir 'your/path/syn_data' \
--output-dir /your/path/save \

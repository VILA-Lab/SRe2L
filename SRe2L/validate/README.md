# Train models on relabeled data

## Preparation

- Python >= 3.8
- PyTorch >= 2.0.0
- Torchvision >= 0.15.1
- Modify PyTorch source code `torch.utils.data._utils.fetch._MapDatasetFetcher` to support *multi-processing loading* of soft label data and mix configurations.
  ```python
  ### Original code
  class _MapDatasetFetcher(_BaseDatasetFetcher):
      def fetch(self, possibly_batched_index):
          if self.auto_collation:
              if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                  data = self.dataset.__getitems__(possibly_batched_index)
              else:
                  data = [self.dataset[idx] for idx in possibly_batched_index]
          else:
              data = self.dataset[possibly_batched_index]
          return self.collate_fn(data)

  ### Modified code
  class _MapDatasetFetcher(_BaseDatasetFetcher):
      def fetch(self, possibly_batched_index):
          if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
              mix_index, mix_lam, mix_bbox, soft_label = self.dataset.load_batch_config(possibly_batched_index[0])

          if self.auto_collation:
              if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                  data = self.dataset.__getitems__(possibly_batched_index)
              else:
                  data = [self.dataset[idx] for idx in possibly_batched_index]
          else:
              data = self.dataset[possibly_batched_index]

          if hasattr(self.dataset, "mode") and self.dataset.mode == 'fkd_load':
              return self.collate_fn(data), mix_index.cpu(), mix_lam, mix_bbox, soft_label.cpu()
          else:
              return self.collate_fn(data)
  ```

## Training

To train a model on relabeled distilled data, run `train.sh` with the desired arguments.

```bash
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
```

Since we modify the PyTorch source code to load the soft labels data and mix configurations before fetching batch data, it will take more memory than the original code due to extra files to be temporarily stored in memory. Thus, we recommend to use a smaller `-j` number of workers to load data and use a larger `--gradient-accumulation-steps` to reduce the memory in model inference. For reference, we use `-j 4 --gradient-accumulation-steps 4` in single RTX 4090 with 24GB memory, `-j 8 --gradient-accumulation-steps 1` in single Tesla A100 with 40GB. There is no effect on `val_loader`, whose `num_workers` can be set to a larger number.

In terms of the FKD-related arguments, they should align to the setting in [relabel](../relabel). For example, `--batch-size` should be the same value in [relabel](../relabel) and `--epochs` argument should be no more than the epochs in [relabel](../relabel).

In terms of the `--gradient-accumulation-steps` argument, it will split the loaded batch data of `--batch-size` into some smaller batch data. For example, if `--gradient-accumulation-steps 4`, it will split the loaded batch data of 1024 into 4 smaller batch data of 256 each. Then, it will accumulate the gradients of 4 smaller batch data and update the model parameters once. In this way, it can reduce the memory in model inference.

In terms of `wandb`, we use it to record the training process. If you don't want to use it, you can set `wandb disabled` in `train.sh`. If you want to use it, you need to set `wandb enabled \\ wandb online` and `--wandb-api-key` in `train.sh`.

## Usage

```
usage: train_FKD.py

[-h] [--batch-size BATCH_SIZE] [--gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS] [--start-epoch START_EPOCH] [--epochs EPOCHS] [-j WORKERS] [--train-dir TRAIN_DIR] [--val-dir VAL_DIR] [--output-dir OUTPUT_DIR] [--cos] [--sgd] [-lr LEARNING_RATE] [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY] [--adamw-lr ADAMW_LR] [--adamw-weight-decay ADAMW_WEIGHT_DECAY] [--model MODEL] [--keep-topk KEEP_TOPK] [-T TEMPERATURE] [--fkd-path FKD_PATH] [--wandb-project WANDB_PROJECT] [--mix-type {mixup,cutmix,None}] [--fkd_seed FKD_SEED]

arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size
  --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS
                        gradient accumulation steps for small gpu memory
  --start-epoch START_EPOCH
                        start epoch
  --epochs EPOCHS       total epoch
  -j WORKERS, --workers WORKERS
                        number of data loading workers
  --train-dir TRAIN_DIR
                        path to training dataset
  --val-dir VAL_DIR     path to validation dataset
  --output-dir OUTPUT_DIR
                        path to output dir
  --cos                 cosine lr scheduler
  --sgd                 sgd optimizer
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        sgd init learning rate
  --momentum MOMENTUM   sgd momentum
  --weight-decay WEIGHT_DECAY
                        sgd weight decay
  --adamw-lr ADAMW_LR   adamw learning rate
  --adamw-weight-decay ADAMW_WEIGHT_DECAY
                        adamw weight decay
  --model MODEL         student model name
  --keep-topk KEEP_TOPK
                        keep topk logits for kd loss
  -T TEMPERATURE, --temperature TEMPERATURE
                        temperature for distillation loss
  --fkd-path FKD_PATH   path to fkd label
  --wandb-project WANDB_PROJECT
                        wandb project name
  --wandb-api-key WANDB_API_KEY
                        wandb api key
  --mix-type {mixup,cutmix,None}
                        mixup or cutmix or None
  --fkd_seed FKD_SEED   seed for batch loading sampler
```

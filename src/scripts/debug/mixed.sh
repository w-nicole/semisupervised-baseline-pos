#!/bin/bash

# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

save_path=${1:-"./experiments/debug/self_train/mixed"}
train_languages="English Dutch"
val_languages="English Dutch"
data_path=${2:-"../../ud-treebanks-v1.4"}
checkpoint_1=${3:-"../../alt/semisupervised-baseline-pos/experiments/subset/masked/subset_count=10/version_x3do118v/ckpts/ckpts_epoch=1-val_English_pos_acc_epoch=65.902.ckpt"}
checkpoint_2=${4:-"../../alt/semisupervised-baseline-pos/experiments/subset/unmasked/subset_count=10/version_3xr3fjl9/ckpts/ckpts_epoch=15-val_English_pos_acc_epoch=81.096.ckpt"}
bs=32
ep=1

python3 src/train_self_training.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --max_epochs $ep \
    --subset_ratio 0.01 \
    --warmup_portion 0.1 \
    --view_checkpoint_1 $checkpoint_1 \
    --view_checkpoint_2 $checkpoint_2 \
    --is_masked_view_1 "y" \
    --is_masked_view_2 "n" \
    --group "debug" \
    --job_type "self_train" \
    --name "mixed" \
    --default_save_path "$save_path" \
    --freeze_mbert "y" \
    --use_subset_complement "y" \
    --gpus 1
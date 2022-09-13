#!/bin/bash

# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

save_path=${1:-"./experiments/debug/self_train/pure"}
train_languages="English Dutch"
val_languages="English Dutch"
data_path=${2:-"../../ud-treebanks-v1.4"}
checkpoint_1=${3:-"../../alt/semisupervised-baseline-pos/experiments/subset/unmasked/subset_count=10/version_3xr3fjl9/ckpts/ckpts_epoch=15-val_English_pos_acc_epoch=81.096.ckpt"}
checkpoint_2=${3:-"../../alt/semisupervised-baseline-pos/experiments/subset/unmasked_alt_seed/subset_count=10/version_w7soscl3/ckpts/ckpts_epoch=11-val_English_pos_acc_epoch=79.605.ckpt"}

bs=32
ep=1

python3 src/train_self_training.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --view_checkpoint_1 $checkpoint_1 \
    --view_checkpoint_2 $checkpoint_2 \
    --is_masked_view_1 "n" \
    --is_masked_view_2 "n" \
    --group "debug" \
    --job_type "self_train" \
    --name "pure" \
    --default_save_path "$save_path" \
    --freeze_mbert "y" \
    --gpus 1
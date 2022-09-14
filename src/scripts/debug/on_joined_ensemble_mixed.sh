#!/bin/bash

# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

save_path=${1:-"./experiments/debug/self_train/mixed"}
train_languages="English"
val_languages="English"
data_path=${2:-"../../ud-treebanks-v1.4"}
joined_view_checkpoints=('a'
    'b'
    'c')
bs=32
ep=1

python3 src/train_on_joined_ensemble.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --joined_view_checkpoint $joined_view_checkpoint \
    --is_masked_view_1 "y" \
    --is_masked_view_2 "n" \
    --group "debug" \
    --job_type "on_joined_ensemble" \
    --name "mixed" \
    --default_save_path "$save_path" \
    --freeze_mbert "y" \
    --gpus 1
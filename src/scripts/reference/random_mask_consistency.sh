#!/bin/bash

# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

save_path=${1:-"./experiments/reference/random_mask/english_subset_fixed"}
train_languages="English"
val_languages="English"
data_path=${2:-"../../ud-treebanks-v1.4"}

bs=32
ep=1

python3 src/train_encoder.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --max_epochs $ep \
    --subset_ratio 0.01 \
    --warmup_portion 0.1 \
    --mask_probability 0 \
    --group "reference" \
    --job_type "random_mask" \
    --name "english_subset_fixed" \
    --default_save_path "$save_path" \
    --freeze_mbert "y" \
    --gpu 1
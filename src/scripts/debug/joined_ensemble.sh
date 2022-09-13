#!/bin/bash

# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

save_path=${1:-"./experiments/debug/self_train/mask_prob_0.01"}
train_languages="English"
val_languages="English"
data_path=${2:-"../../ud-treebanks-v1.4"}
bs=32
ep=1

python3 src/train_joined_ensemble.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --subset_ratio 0.01 \
    --batch_size $bs \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --mask_probability 0.01 \
    --unraveled_predictions "y" \
    --double_pass "y" \
    --group "debug" \
    --job_type "mask_prob" \
    --name "p=0.01" \
    --default_save_path "$save_path" \
    --freeze_mbert "y" \
    --gpus 1
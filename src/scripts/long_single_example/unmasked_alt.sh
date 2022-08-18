#!/bin/bash

# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

save_path=${1:-"./experiments/subset_explore/long_single_example/unmasked_alt_seed/english"}
train_languages="English"
val_languages="English"
data_path=${2:-"../../ud-treebanks-v1.4"}

bs=32
ep=100

python3 src/train_encoder.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --default_save_path "$save_path" \
    --freeze_mbert "n" \
    --subset_seed 42 \
    --gpu 1 \
    --schedule "reduceOnPlateau" \
    --default_learning_rate 5e-3 \
    --group "subset_explore" \
    --job_type "long_single_example" \
    --masked "n" \
    --name "unmasked_alt_seed" \
    --seed 0


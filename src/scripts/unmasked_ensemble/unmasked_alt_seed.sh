#!/bin/bash

# Adapted from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

save_path=${1:-"./experiments/unmasked_ensemble/unmasked"}
train_languages="English"
val_languages="English"
data_path=${2:-"../../ud-treebanks-v1.4"}

bs=32
ep=20

python3 src/train_encoder.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --max_epochs $ep \
    --subset_count 10 \
    --seed 0 \
    --subset_seed 42 \
    --hyperparameter_names "subset_count" \
    --warmup_portion 0.1 \
    --mask_probability 0 \
    --group "random_mask" \
    --job_type "unmasked_ensemble" \
    --name "unmasked_alt_seed" \
    --default_save_path "$save_path" \
    --freeze_mbert "n" \
    --schedule "reduceOnPlateau" \
    --weight_decay 0.01 \
    --mbert_learning_rate 5e-5 \
    --default_learning_rate 5e-3 \
    --gpu 1
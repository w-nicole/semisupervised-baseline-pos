#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changing, removing)
# Moved some arguments to default arguments.
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

save_path=${1:-"./experiments/compare/subset/fixed"}
train_languages="English"
val_languages="English"
data_path=${2:-"../../ud-treebanks-v1.4"}

bs=32
ep=50

python3 src/train_latent_base.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --latent_size 64 \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --group "compare" \
    --job_type "for_subset" \
    --name "latent_64_fixed" \
    --mse_weight 0 \
    --default_save_path "$save_path" \
    --freeze_mbert "y" \
    --gpu 1 
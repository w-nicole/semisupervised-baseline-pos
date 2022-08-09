#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changing, removing)
# Moved some arguments to default arguments.
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

save_path=${1:-"./experiments/size_2"}
train_languages="English"
val_languages="English"
data_path=${2:-"../../ud-treebanks-v1.4"}

bs=32
ep=4

python3 src/train_latent_base.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --latent_size 2 \
    --batch_size $bs \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --group "size_2" \
    --job_type "explore" \
    --name "english" \
    --default_save_path "$save_path" \
    --freeze_mbert "n" \
    --mse_weight 0 \
    --gpu 1 
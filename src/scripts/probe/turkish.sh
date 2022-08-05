#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changing, removing)
# Moved some arguments to default arguments.
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

save_path=${1:-"./experiments/probe/turkish"}
train_languages="Turkish"
val_languages="English Dutch Turkish"
data_path=${2:-"../../ud-treebanks-v1.4"}
checkpoint=${3:-""}

bs=16
ep=100

python3 src/train_latent_probe.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --max_epochs $ep \
    --default_learning_rate 5e-3 \
    --warmup_portion 0.1 \
    --group "probe" \
    --latent_space_checkpoint $checkpoint \
    --job_type "validate" \
    --name "turkish" \
    --default_save_path "$save_path" \
    --freeze_mbert "n" \
    --gpu 1 
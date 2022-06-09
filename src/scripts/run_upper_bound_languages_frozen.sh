#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory where this is run
# Simplified `example/surprising-mbert/evaluate.sh` script due to no need for non-POS evaluation by removing irrelevant code, options, etc.
# Added hyperparameters that yield best performance on English val for last layer finetuned BERT.
# Edited arguments.
# Edited to run all languages as an upper bound instead.

seed=${1:-42}
model=${2:-"bert-base-multilingual-cased"}
task=${3:-"udpos"}

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)

save_path=${4:-"./experiments/encoder_upper_bounds/frozen_total"}

langs=(English)
data_path=${5:-"../ud-treebanks-v1.4"}

bs=16
ep=20
lr=5e-5

langs="German Dutch English"

for src in $langs; do
    python3 src/train_encoder.py \
        --seed "$seed" \
        --task "$task" \
        --data_dir "$data_path" \
        --trn_langs $src \
        --val_langs $src \
        --pretrain "$model" \
        --batch_size $bs \
        --learning_rate $lr \
        --max_epochs $ep \
        --warmup_portion 0.1 \
        --freeze_layer 12 \
        --concat_all_hidden_states "y" \
        --default_save_path "$save_path" \
        --exp_name $src \
        --gpus 1
    done
    

#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory where this is run
# Simplified `example/surprising-mbert/evaluate.sh` script due to no need for non-POS evaluation by removing irrelevant code, options, etc.
# Added hyperparameters that yield best performance on English val for last layer finetuned BERT.
# Edited arguments.
# Edited to run the train_encoder script instead.

seed=${1:-42}
model=${2:-"bert-base-multilingual-cased"}
task=${3:-"udpos"}

bs=16
ep=3
lr=5e-5
weight_decay=0.1

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)

save_path=${4:-"./experiments/weight_decay/$weight_decay"}

langs=(English)
data_path=${5:-"../ud-treebanks-v1.4"}

python3 src/train_encoder.py \
    --seed "$seed" \
    --task "$task" \
    --data_dir "$data_path" \
    --trn_langs "${langs[@]}" \
    --val_langs "${langs[@]}" \
    --pretrain "$model" \
    --batch_size $bs \
    --learning_rate $lr \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --default_save_path "$save_path" \
    --wandb_group "weight_decay" \
    --wandb_job_type "search" \
    --exp_name "linear" \
    --freeze_mbert "n" \
    --gpus 1 \
    --weight_decay $weight_decay
    
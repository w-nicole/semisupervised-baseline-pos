#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3
# Simplified `example/surprising-mbert/evaluate.sh` script due to no need for non-POS evaluation by removing irrelevant code, options, etc.
# Substitution of parameters that yield best val English on seed 42.
# Addition of (sometimes commented out) parameters for faster development, such as subset ratio.
# Changed root directory for run.

seed=${1:-42}
model=${2:-"bert-base-multilingual-cased"}
task=${3:-"udpos"}
freeze=${4:-"-1"}

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)

save_path=${5:-"../experiments/their_parameters_single_val_run"}

src="English"
tgt=(Bulgarian Danish German English Spanish Persian Hungarian Italian Dutch Polish Portuguese Romanian Slovak Slovenian Swedish)
data_path=${6:-"../../ud-treebanks-v1.4"}

for bs in 16 32; do
    for lr in 2e-5 3e-5 5e-5; do
        for ep in 3 4; do
            python3 train.py \
                --seed "$seed" \
                --task "$task" \
                --data_dir "$data_path" \
                --trn_langs $src \
                --val_langs $src \
                --tst_langs "${tgt[@]}" \
                --pretrain "$model" \
                --batch_size $bs \
                --learning_rate $lr \
                --max_epochs $ep \
                --warmup_portion 0.1 \
                --freeze_layer "$freeze" \
                --default_save_path "$save_path"/"$task"/0-shot-finetune-freeze"$freeze"/"$model_name" \
                --exp_name bs$bs-lr$lr-ep$ep \
                --subset_ratio=0.0001 \
                --gpus 1 \
        done
    done
done

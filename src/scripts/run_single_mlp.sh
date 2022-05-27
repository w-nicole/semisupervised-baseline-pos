#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory where this is run
# Simplified `example/surprising-mbert/evaluate.sh` script due to no need for non-POS evaluation by removing irrelevant code, options, etc.
# Edited arguments. 
# Uses best English val params.

seed=${1:-42}
model=${2:-"bert-base-multilingual-cased"}
task=${3:-"udpos"}
freeze=${4:-"12"}

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)

save_path=${5:-"./experiments/single_mlp"}

src="English"
tgt=(Bulgarian Danish German English Spanish Persian Hungarian Italian Dutch Polish Portuguese Romanian Slovak Slovenian Swedish)
data_path=${6:-"../ud-treebanks-v1.4"}

for hidden_size in 32 256 512; do
    python3 src/train.py \
        --seed "$seed" \
        --task "$task" \
        --data_dir "$data_path" \
        --trn_langs $src \
        --val_langs $src \
        --tst_langs "${tgt[@]}" \
        --pretrain "$model" \
        --batch_size 16 \
        --learning_rate 5e-5 \
        --max_epochs 3 \
        --warmup_portion 0.1 \
        --default_save_path "$save_path"/"$task"/0-shot-finetune-freeze"$freeze"/"$model_name" \
        --exp_name mlp_hidden_size$hidden_size \
        --use_hidden_layer "y" \
        --hidden_layer_size $hidden_size \
        --gpus 1
done

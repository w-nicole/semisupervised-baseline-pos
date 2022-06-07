#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3
# Simplified `example/surprising-mbert/evaluate.sh` script due to no need for non-POS evaluation by removing irrelevant code, options, etc.
# Substitution of parameters that yield best val English on seed 42, frozen, last layer.
# Addition of (sometimes commented out) parameters for faster development, such as subset ratio.
# Changed root directory for run.
# Changed save name for path.
# Validate on everything instead.
# Added model freeze and layer search instead of original grid search, as well as extra run for concat.

seed=${1:-42}
model=${2:-"bert-base-multilingual-cased"}
task=${3:-"udpos"}
freeze=${4:-"12"} # This only affects the naming

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)

save_path=${5:-"./experiments/layer_search_frozen"}

src="English"
tgt=(Bulgarian Danish German English Spanish Persian Hungarian Italian Dutch Polish Portuguese Romanian Slovak Slovenian Swedish)
data_path=${6:-"../ud-treebanks-v1.4"}

bs=16
lr=5e-5
ep=4

python3 src/train.py \
    --seed "$seed" \
    --task "$task" \
    --data_dir "$data_path" \
    --trn_langs $src \
    --val_langs "${tgt[@]}" \
    --tst_langs "${tgt[@]}" \
    --pretrain "$model" \
    --batch_size $bs \
    --learning_rate $lr \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --freeze_layer 12 \
    --concat_all_hidden_states "y" \
    --default_save_path "$save_path"/"$task"/0-shot-finetune-layer"concat"/"$model_name" \
    --exp_name bs$bs-lr$lr-ep$ep \
    --gpus 1
                
for layer in {1..12}; do
            python3 src/train.py \
                --seed "$seed" \
                --task "$task" \
                --data_dir "$data_path" \
                --trn_langs $src \
                --val_langs "${tgt[@]}" \
                --tst_langs "${tgt[@]}" \
                --pretrain "$model" \
                --batch_size $bs \
                --learning_rate $lr \
                --max_epochs $ep \
                --warmup_portion 0.1 \
                --freeze_layer 12 \
                --feature_layer "$layer" \
                --default_save_path "$save_path"/"$task"/0-shot-finetune-layer"$layer"/"$model_name" \
                --exp_name bs$bs-lr$lr-ep$ep \
                --gpus 1
        done
    done
done


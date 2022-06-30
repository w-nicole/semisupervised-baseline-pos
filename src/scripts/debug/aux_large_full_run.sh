#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory where this is run
# Simplified `example/surprising-mbert/evaluate.sh` script due to no need for non-POS evaluation by removing irrelevant code, variables, options, etc.
# Added hyperparameters that yield best performance on English val for last layer finetuned BERT.
# Edited arguments.
# Edited to run the train_decoder script instead.

seed=${1:-42}
model=${2:-"bert-base-multilingual-cased"}
task=${3:-"udpos"}

model_name=$(echo "$model" | tr '/' '\n' | tail -n1)

save_path=${4:-"./experiments/normal/large_aux/phase_3"}

src="English Dutch"
tgt="English Dutch"
data_path=${5:-"../ud-treebanks-v1.4"}
decoder_checkpoint=${6:-"./experiments/normal/large/auxiliary/phase_2_auxiliary_l_1_h_32_s_64/version_2/ckpts/ckpts_epoch=5-val_English_decoder_loss=65.581.ckpt"}

bs=128
ep=15
lr=1e-3

python3 src/train_decoder.py \
    --seed "$seed" \
    --task "$task" \
    --data_dir "$data_path" \
    --trn_langs $src \
    --val_langs $tgt \
    --pretrain "$model" \
    --batch_size $bs \
    --learning_rate $lr \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --default_save_path "$save_path" \
    --exp_name lr_"$lr"_bs"$bs" \
    --gpus 1 \
    --patience $ep \
    --decoder_checkpoint "$decoder_checkpoint" \
    --schedule "reduceOnPlateau" \
    --prior_type "optimized_data"

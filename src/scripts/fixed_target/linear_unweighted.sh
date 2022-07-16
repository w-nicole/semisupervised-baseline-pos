#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changing, removing)
# Moved some arguments to default arguments.
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

save_path=${1:-"./experiments/fixed_target/pretrained/lstm/unweighted_fcn"}
train_languages="English Dutch"
val_languages="English Dutch"
data_path=${2:-"../ud-treebanks-v1.4"}
mbert_checkpoint=${3:-"./experiments/components/mbert_pretrained/linear/version_1ylqtlld/ckpts/ckpts_epoch=2-val_English_acc_epoch_monitor=96.807.ckpt"}

bs=16
ep=3
lr=5e-3
latent_size=64
pos_hidden_layers=1
pos_hidden_size=$latent_size
reconstruction_hidden_size=$latent_size
reconstruction_hidden_layers=1

python3 src/train_latent_to_pos_cross.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --batch_size $bs \
    --learning_rate $lr \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --default_save_path "$save_path" \
    --pos_hidden_layers $pos_hidden_layers \
    --pos_hidden_size $pos_hidden_size \
    --reconstruction_hidden_layers $reconstruction_hidden_layers \
    --reconstruction_hidden_size $reconstruction_hidden_size \
    --encoder_mbert_checkpoint $mbert_checkpoint \
    --target_mbert_checkpoint $mbert_checkpoint \
    --reconstruction_model_type "fcn" \
    --pos_model_type "fcn" \
    --gpus 1 \
    --freeze_mbert "y" \
    --schedule "reduceOnPlateau"
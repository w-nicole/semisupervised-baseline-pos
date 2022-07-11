#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changing, removing)
# Moved some arguments to default arguments.
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

save_path=${1:-"./experiments/debug/mse/by_index_small_big_model"}
train_languages="Dutch"
val_languages="Dutch"
data_path=${2:-"../ud-debug-small"}
mbert_checkpoint=${3:-"./experiments/components/mbert_pretrained/linear/version_2f3wjdti/ckpts/ckpts_epoch=2-val_English_acc_epoch_monitor=96.807.ckpt"}

bs=16
ep=10000
lr=1e-3
latent_size=512

python3 src/train_latent_to_pos.py \
    --data_dir "$data_path" \
    --trn_langs $train_languages \
    --val_langs $val_languages \
    --latent_size $latent_size \
    --batch_size $bs \
    --latent_kl_weight 0 \
    --pos_nll_weight 0 \
    --learning_rate $lr \
    --max_epochs $ep \
    --warmup_portion 0.01 \
    --default_save_path "$save_path" \
    --exp_name decoder_pos_"$pos_hidden_layers","$pos_hidden_size"_mbert_"$mbert_hidden_layers","$mbert_hidden_size" \
    --encoder_mbert_checkpoint $mbert_checkpoint \
    --freeze_mbert "y" \
    --gpus 1 \
    --sampling "n" \
    --patience $ep \
    --encoder_hidden_layers 4 \
    --encoder_hidden_size $latent_size \
    --decoder_reconstruction_hidden_layers 4 \
    --decoder_reconstruction_hidden_size $latent_size \
    --decoder_reconstruction_nonlinear_first "n" \
    --softmax_before_reconstruction "y"
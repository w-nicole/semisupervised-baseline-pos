#!/bin/bash

# Taken from Shijie Wu's crosslingual-nlp repository.
# See LICENSE in this codebase for license information.

# Changes made relative to original:
# Changed to python3, directory relationship, correct script to run
# Edited arguments and hyperparameters (adding, changing, removing)
# Moved some arguments to default arguments.
# Changed to not have source/target but train/val languages.
# Simplified `example/surprising-mbert/evaluate.sh` script to remove irrelevant code.

save_path=${1:-"./experiments/baseline/Dutch/mbert_fixed/encoder_pretrained/decoder_random/kl_0.1"}
train_languages="Dutch"
val_languages="English Dutch"
data_path=${2:-"../ud-treebanks-v1.4"}
mbert_checkpoint=${3:-""}
encoder_checkpoint=${4:-"./experiments/components/mbert_fixed/encoder/linear/version_0/ckpts/ckpts_epoch=3-val_English_acc=95.332.ckpt"}
decoder_checkpoint=${5:-""}
bs=16
ep=2
lr=0.005

python3 src/train_decoder.py \
	--data_dir "$data_path" \
	--trn_langs $train_languages \
	--val_langs $val_languages \
	--batch_size $bs \
	--learning_rate $lr \
	--max_epochs $ep \
	--warmup_portion 0.1 \
	--default_save_path "$save_path" \
	--exp_name linear \
	--gpus 1 \
	--freeze_mbert "y" \
	--concat_all_hidden_states "y" \
	--encoder_checkpoint $encoder_checkpoint \
	--pos_kl_weight 0.1 \
	--schedule "reduceOnPlateau" \
	--prior_type "optimized_data"

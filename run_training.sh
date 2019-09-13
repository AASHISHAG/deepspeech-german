#!/usr/bin/env bash

deepspeech_path=$1
alphabet_path=$2
exp_path=$3

current_dir=$(pwd)
cd $deepspeech_path

if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/native_client

python -u DeepSpeech.py \
  --train_files $exp_path/data/train.csv \
  --dev_files $exp_path/data/dev.csv \
  --test_files $exp_path/data/test.csv \
  --train_batch_size 12 \
  --dev_batch_size 12 \
  --test_batch_size 12 \
  --n_hidden 375 \
  --epoch 50 \
  --display_step 0 \
  --validation_step 1 \
  --early_stop True \
  --earlystop_nsteps 6 \
  --estop_mean_thresh 0.1 \
  --estop_std_thresh 0.1 \
  --dropout_rate 0.22 \
  --learning_rate 0.00095 \
  --report_count 10 \
  --use_seq_length False \
  --coord_port 8686 \
  --export_dir $exp_path/model_export/ \
  --checkpoint_dir $exp_path/checkpoints/ \
  --decoder_library_path native_client/libctc_decoder_with_kenlm.so \
  --alphabet_config_path $alphabet_path \
  --lm_binary_path $exp_path/lm.binary \
  --lm_trie_path $exp_path/trie \
  "$@"

cd $current_dir

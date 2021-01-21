#!/bin/sh
# MAINTAINER: Aashish Agarwal

for i in 0.000001 0.00001 0.0001 0.001 0.01
do
  rm -rf deepspeech_optimizer/summaries
  rm -rf deepspeech_optimizer/checkpoints
  python DeepSpeech/DeepSpeech.py --train_files "${1}"/train.csv \
  --dev_files "${1}"/dev.csv --test_files "${1}"/test.csv \
  --alphabet_config_path data/alphabet.txt --scorer german-text-corpus/kenlm.scorer \
  --test_batch_size 36 --train_batch_size 24 --dev_batch_size 36 --epochs 75 --learning_rate $i --dropout_rate 0.25 --export_dir models \
  --save_checkpoint_dir deepspeech_optimizer/checkpoints \
  --summary_dir deepspeech_optimizer/summaries
done

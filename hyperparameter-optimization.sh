#!/bin/sh
# MAINTAINER: Aashish Agarwal

for i in 0.000001 0.00001 0.0001 0.001 0.01
do
  ./DeepSpeech.py --train_files ../german-speech-corpus/data_tuda+voxforge+mozilla/sequential/train_100.csv --dev_files ../german-speech-corpus/data_tuda+voxforge+mozilla/sequential/dev.csv --test_files ../german-speech-corpus/data_tuda+voxforge+mozilla/sequential/test.csv --alphabet_config_path ../deepspeech-german/data/alphabet.txt --lm_trie_path ../dataset-german/trie --lm_binary_path ../dataset-german/lm.binary --test_batch_size 36 --train_batch_size 24 --dev_batch_size 36 --epochs 75 --learning_rate $i --dropout_rate 0.25 --export_dir ../models
  rm -rf ../.local/share/deepspeech/summaries
  rm -rf ../.local/share/deepspeech/checkpoints
done

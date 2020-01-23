#!/bin/sh
# MAINTAINER: Aashish Agarwal

source ../python-environments/env/bin/activate

./DeepSpeech.py --train_files ../german-speech-corpus/data_tuda+voxforge/train.csv --dev_files ../german-speech-corpus/data_tuda+voxforge/dev.csv --test_files ../german-speech-corpus/data_tuda+voxforge/test.csv --alphabet_config_path ../deepspeech-german/data/alphabet.txt --lm_trie_path ../dataset-german/trie --lm_binary_path ../dataset-german/lm.binary --test_batch_size 36 --train_batch_size 24 --dev_batch_size 36 --epochs 75 --learning_rate 0.0005 --dropout_rate 0.40 --export_dir ../models --checkpoint_dir ../checkpoints

#!/bin/bash
# MAINTAINER: Aashish Agarwal

source python-environments/env/bin/activate
export PYTHONPATH=./DeepSpeech/training
python DeepSpeech/DeepSpeech.py --test_files "${1}"/test.csv --alphabet_config_path data/alphabet.txt \
--scorer german-text-corpus/kenlm.scorer --test_batch_size 36 --train_batch_size 24 --dev_batch_size 36 \
--epochs 75 --learning_rate 0.0001 --dropout_rate 0.30 --export_dir models

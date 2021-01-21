#!/bin/bash
# MAINTAINER: Aashish Agarwal

source python-environments/env/bin/activate
export PYTHONPATH=./DeepSpeech/training

ALPHABET=${2:-data/alphabet.txt}
SCORER=${3:-german-text-corpus/kenlm.scorer}
PROCESSING_DIR=${4:-deepspeech_processing}
MODELS=${5:-models}

python DeepSpeech/DeepSpeech.py --train_files "${1}"/train.csv --dev_files "${1}"/dev.csv \
--test_files "${1}"/test.csv --alphabet_config_path "${ALPHABET}" \
--scorer "${SCORER}" --test_batch_size 36 --train_batch_size 24 --dev_batch_size 36 --epochs 5 --learning_rate 0.0005 --dropout_rate 0.40 \
--export_dir "${MODELS}" --checkpoint_dir "${PROCESSING_DIR}" --summary_dir "${PROCESSING_DIR}"/summaries \
 "${6}"
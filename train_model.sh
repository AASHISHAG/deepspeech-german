#!/bin/bash
# MAINTAINER: Aashish Agarwal

source python-environments/bin/activate
export PYTHONPATH=./DeepSpeech/training

ALPHABET=${2:-data/alphabet.txt}
SCORER=${3:-german-text-corpus/kenlm.scorer}
PROCESSING_DIR=${4:-deepspeech_processing}
MODELS=${5:-models}

rm -rf deepspeech_processing/summaries
rm -rf deepspeech_processing/checkpoints
python DeepSpeech/DeepSpeech.py --train_files "${1}"/train.csv \
--dev_files "${1}"/dev.csv \
--test_files "${1}"/test.csv \
--alphabet_config_path "${ALPHABET}" \
--scorer "${SCORER}" \
--test_batch_size 36 --train_batch_size 24 --dev_batch_size 36 \
--export_language "de-Latn-DE" --export_license "Apache-2.0" --export_model_name "DeepSpeech German" \
--export_model_version "0.0.5" --export_author_id "deepspeech-german" \
--epochs 75 --learning_rate 0.0005 --dropout_rate 0.40 --export_dir "${MODELS}" -v 1 \
--checkpoint_dir "${PROCESSING_DIR}"/checkpoints --summary_dir "${PROCESSING_DIR}"/summaries
#!/bin/bash
# MAINTAINER: Aashish Agarwal

source python-environments/bin/activate
export PYTHONPATH=./DeepSpeech/training

ALPHABET=${3:-data/alphabet.txt}
SCORER=${4:-german-text-corpus/kenlm.scorer}
PROCESSING_DIR=${5:-deepspeech_transfer_processing}
MODELS=${6:-transfer_models}
DROP_LAYERS=${7:-1}

mkdir "${PROCESSING_DIR}"
mkdir "${PROCESSING_DIR}"/checkpoints
cp -R "${2}"/* "${PROCESSING_DIR}"/checkpoints/
python DeepSpeech/DeepSpeech.py --train_files "${1}"/train.csv --dev_files "${1}"/dev.csv \
--test_files "${1}"/test.csv --alphabet_config_path "${ALPHABET}" \
--scorer "${SCORER}" --test_batch_size 36 --train_batch_size 24 --dev_batch_size 36 --epochs 75 --learning_rate 0.0005 --dropout_rate 0.40 \
--export_dir "${MODELS}" --drop_source_layers "${DROP_LAYERS}" \
--export_language "de-Latn-DE" --export_license "Apache-2.0" --export_model_name "DeepSpeech German" \
--export_model_version "0.0.5" --export_author_id "deepspeech-german" \
--checkpoint_dir "${PROCESSING_DIR}"/checkpoints --summary_dir "${PROCESSING_DIR}"/summaries "$8"

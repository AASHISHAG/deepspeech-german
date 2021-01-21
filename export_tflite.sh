#!/bin/bash

source python-environments/bin/activate
export PYTHONPATH=./DeepSpeech/training

ALPHABET=${1:-data/alphabet.txt}
PROCESSING_DIR=${2:-deepspeech_processing}
MODELS=${3:-models}

python DeepSpeech/DeepSpeech.py --export_tflite --export_dir "${MODELS}" --alphabet_config_path "${ALPHABET}" \
--export_language "de-Latn-DE" --export_license "Apache-2.0" --export_model_name "DeepSpeech German" \
--export_model_version "0.0.5" --export_author_id "deepspeech-german" \
--checkpoint_dir "${PROCESSING_DIR}"/checkpoints


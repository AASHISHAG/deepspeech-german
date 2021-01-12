#!/bin/bash

source python-environments/bin/activate
export PYTHONPATH=./DeepSpeech/training
ARCH=${1:-cpu}
python DeepSpeech/data/lm/generate_lm.py --input_txt german-text-corpus/clean_vocab.txt --output_dir german-text-corpus --top_k 500000 --kenlm_bins kenlm/build/bin/ --arpa_order 3 --max_arpa_memory "80%" --binary_a_bits 255 --binary_q_bits 8 --arpa_prune "0" --binary_type trie
python DeepSpeech/util/taskcluster.py --target DeepSpeechNativeClient --arch "${ARCH}"
DeepSpeechNativeClient/generate_scorer_package --alphabet data/alphabet.txt --lm german-text-corpus/lm.binary --vocab german-text-corpus/clean_vocab.txt \
  --package german-text-corpus/kenlm.scorer --default_alpha 0.931289039105002 --default_beta 1.1834137581510284


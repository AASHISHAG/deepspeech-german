#! /usr/bin/env python

"""
1. Load all corpora where a path is given.
2. Clean transcriptions.
3. Merge all corpora
4. Create Train/Dev/Test splits
5. Export for DeepSpeech
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, os.path.pardir)))

import argparse

import audiomate
from audiomate.corpus import io
from audiomate.corpus import subset

import text_cleaning


def clean_transcriptions(corpus):
    for utterance in corpus.utterances.values():
        ll = utterance.label_lists[audiomate.corpus.LL_WORD_TRANSCRIPT]

        for label in ll:
            label.value = text_cleaning.clean_sentence(label.value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for training.')
    parser.add_argument('target_path', type=str)
    parser.add_argument('--tuda', type=str)
    parser.add_argument('--voxforge', type=str)
    parser.add_argument('--swc', type=str)
    parser.add_argument('--mailabs', type=str)
    parser.add_argument('--cv', type=str)

    args = parser.parse_args()

    tuda_path = args.tuda
    voxforge_path = args.voxforge
    swc_path = args.swc
    mailabs_path = args.mailabs
    cv_path = args.cv

    corpora = []

    if tuda_path is not None:
        tuda_corpus = audiomate.Corpus.load(tuda_path, reader='tuda')
        corpora.append(tuda_corpus)

    if voxforge_path is not None:
        voxforge_corpus = audiomate.Corpus.load(
            voxforge_path, reader='voxforge')
        corpora.append(voxforge_corpus)

    if swc_path is not None:
        swc_corpus = audiomate.Corpus.load(swc_path, reader='kaldi')
        corpora.append(swc_corpus)

    if mailabs_path is not None:
        mailabs_corpus = audiomate.Corpus.load(mailabs_path, reader='mailabs')
        corpora.append(mailabs_corpus)

    if cv_path is not None:
        cv_corpus = audiomate.Corpus.load(cv_path, reader='common-voice')
        corpora.append(cv_corpus)

    if len(corpora) <= 0:
        raise ValueError('No Corpus given!')

    merged_corpus = audiomate.Corpus.merge_corpora(corpora)
    clean_transcriptions(merged_corpus)

    splitter = subset.Splitter(merged_corpus, random_seed=38)
    splits = splitter.split_by_length_of_utterances(
        {'train': 0.7, 'dev': 0.15, 'test': 0.15}, separate_issuers=True)

    merged_corpus.import_subview('train', splits['train'])
    merged_corpus.import_subview('dev', splits['dev'])
    merged_corpus.import_subview('test', splits['test'])

    deepspeech_writer = io.MozillaDeepSpeechWriter()
    deepspeech_writer.save(merged_corpus, args.target_path)

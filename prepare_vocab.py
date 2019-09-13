#! /usr/bin/env python

"""
1. Load text corpus
2. Clean text
3. Extend with transcriptions from training data
4. Save
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, os.path.pardir)))

import argparse
from audiomate.utils import textfile

import text_cleaning


def read_training_transcripts(path):
    transcripts = []

    for entry in textfile.read_separated_lines_generator(path, separator=',', max_columns=3,
                                                         ignore_lines_starting_with=['wav_filename']):
        transcripts.append(entry[2])

    return transcripts


parser = argparse.ArgumentParser(description='Clean text corpus.')
parser.add_argument('source_path', type=str)
parser.add_argument('target_path', type=str)
parser.add_argument('--training_csv', type=str)

args = parser.parse_args()

index = 0

with open(args.source_path, 'r') as source_file, open(args.target_path, 'w') as target_file:
    for index, line in enumerate(source_file):
        cleaned_sentence = text_cleaning.clean_sentence(line)
        target_file.write('{}\n'.format(cleaned_sentence))

        if index % 1000 == 0:
            print(index)

    print('Cleaned {} lines!'.format(index))

    if args.training_csv is not None:
        training_transcripts = read_training_transcripts(args.training_csv)
        target_file.write('\n'.join(training_transcripts))

        print('Added {} transcripts from training data!'.format(len(training_transcripts)))

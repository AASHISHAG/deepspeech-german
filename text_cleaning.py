# -*- coding: utf-8 -*-

import string
import collections
import re

import num2words

#
#   Number patterns
#
int_pattern = re.compile(r'[0-9]+')
float_pattern = re.compile(r'[0-9]+[,\.][0-9]+')

#
#   Allowed characters a-zA-Z'äüö
#
allowed = list(string.ascii_lowercase)
allowed.append("'")
allowed.append(' ')
allowed.extend(list('äöü'))

#
#   Replacement characters
#
replacer = {
    'àáâãåāăąǟǡǻȁȃȧ': 'a',
    'æǣǽ': 'ä',
    'çćĉċč': 'c',
    'ďđ': 'd',
    'èéêëēĕėęěȅȇȩε': 'e',
    'ĝğġģǥǧǵ': 'g',
    'ĥħȟ': 'h',
    'ìíîïĩīĭįıȉȋ': 'i',
    'ĵǰ': 'j',
    'ķĸǩǩκ': 'k',
    'ĺļľŀł': 'l',
    'м': 'm',
    'ñńņňŉŋǹ': 'n',
    'òóôõøōŏőǫǭǿȍȏðο': 'o',
    'œ': 'ö',
    'ŕŗřȑȓ': 'r',
    'śŝşšș': 's',
    'ţťŧț': 't',
    'ùúûũūŭůűųȕȗ': 'u',
    'ŵ': 'w',
    'ýÿŷ': 'y',
    'źżžȥ': 'z',
    'ß': 'ss',
    '-­': ' '
}

#
#   Various replacement rules
#

special_replacers = {
    ' $ ': 'dollar',
    ' £ ': 'pfund',
    'm³': 'kubikmeter',
    'km²': 'quadratkilometer',
    'm²': 'quadratmeter'
}

replacements = {}
replacements.update(special_replacers)

for all, replacement in replacer.items():
    for to_replace in all:
        replacements[to_replace] = replacement


#
#   Utils
#

def replace_symbols(word):
    """ Apply all replacement characters/rules to the given word. """
    result = word

    for to_replace, replacement in replacements.items():
        result = result.replace(to_replace, replacement)

    return result


def remove_symbols(word):
    """ Remove all symbols that are not allowed. """
    result = word
    bad_characters = []

    for c in result:
        if c not in allowed:
            bad_characters.append(c)

    for c in bad_characters:
        result = result.replace(c, '')

    return result


def word_to_num(word):
    """ Replace numbers with their written representation. """
    result = word

    match = float_pattern.search(result)

    while match is not None:
        num_word = num2words.num2words(float(match.group().replace(',', '.')), lang='de').lower()
        before = result[:match.start()]
        after = result[match.end():]
        result = ' '.join([before, num_word, after])
        match = float_pattern.search(result)

    match = int_pattern.search(result)

    while match is not None:
        num_word = num2words.num2words(int(match.group()), lang='de')
        before = result[:match.start()]
        after = result[match.end():]
        result = ' '.join([before, num_word, after])
        match = int_pattern.search(result)

    return result


def get_bad_character(text):
    """ Return all characters in the text that are not allowed. """
    bad_characters = set()

    for c in text:
        if c not in allowed:
            bad_characters.add(c)

    return bad_characters


def clean_word(word):
    """
    Clean the given word.

    1. numbers to words
    2. character/rule replacements
    3. delete disallowed symbols
    """
    word = word.lower()
    word = word_to_num(word)
    word = replace_symbols(word)
    word = remove_symbols(word)

    bad_chars = get_bad_character(word)

    if len(bad_chars) > 0:
        print('Bad characters in "{}"'.format(word))
        print('--> {}'.format(', '.join(bad_chars)))

    return word


def clean_sentence(sentence):
    """
    Clean the given sentence.

    1. split into words by spaces
    2. numbers to words
    3. character/rule replacements
    4. delete disallowed symbols
    4. join with spaces
    """
    words = sentence.strip().split(' ')
    cleaned_words = []

    for word in words:
        cleaned_word = clean_word(word)
        cleaned_words.append(cleaned_word)

    return ' '.join(cleaned_words)

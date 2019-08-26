""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict
from jamo import h2j, j2h
from jamo.jamo import _jamo_char_to_hcj

from .korean import ALL_SYMBOLS, PAD, EOS

_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
en_symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet

symbols = ALL_SYMBOLS
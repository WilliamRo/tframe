from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class DataConfigs(object):
  train_size = Flag.integer(0, 'Size of training set')
  val_size = Flag.integer(0, 'Size of validation set')
  test_size = Flag.integer(0, 'Size of test set')
  sequence_length = Flag.integer(0, 'Sequence length', is_key=None)
  fixed_length = Flag.boolean(True, 'Whether to fix sequence length.'
                                    'used in AP, etc', is_key=None)
  cheat = Flag.boolean(True, '...', is_key=None)
  multiple = Flag.integer(1, '...', is_key=None)
  noisy = Flag.boolean(None, 'Whether XXX is noisy.', is_key=None)
  prediction_threshold = Flag.float(
    None, 'The prediction threshold used as an parameter for metric function',
    is_key=None)
  permute = Flag.boolean(
    False, 'Whether to permute data (e.g. pMNIST)', is_key=None)
  bits = Flag.integer(
    2, 'Can be used in k-bit temporal order problem', is_key=None)

  test_directly = Flag.boolean(False, 'Whether to use validation set')
  num_words = Flag.integer(0, 'Words number in data set like IMDB', is_key=None)
  max_len = Flag.integer(None, 'Max sequence length. e.g. max_len in IMDB seqs',
                         is_key=None)

  overlap_pct = Flag.float(
    0.0, 'Overlap percent used in converting dataset to `rnn_input`',
    is_key=None)

  random_shift_pct = Flag.float(
    0.0, 'Random shift percent used in converting dataset to `rnn_input`',
    is_key=None)

  train_set = Flag.whatever(None, 'Training set')
  val_set = Flag.whatever(None, 'Validation set')
  test_set = Flag.whatever(None, 'Testing set')


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class DataConfigs(object):
  data_config = Flag.string(None, 'Data set config string', is_key=None)
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

  random_sample_length = Flag.integer(
    None, 'Length of sequences in each batch used in '
          'dataset._convert_to_rnn_input', is_key=None)

  train_set = Flag.whatever(None, 'Training set')
  val_set = Flag.whatever(None, 'Validation set')
  test_set = Flag.whatever(None, 'Testing set')

  class_weights = Flag.list(None, 'Class weights used in weighted '
                                  'cross entropy loss')

  # Financial data configs
  max_level = Flag.integer(None, 'Max level in LOB data', is_key=None)
  volume_only = Flag.boolean(
    None, 'Whether to use volume information only', is_key=None)
  horizon = Flag.integer(None, 'Horizon used in FI-2010 data', is_key=None)

  # BETA configs
  use_wheel = Flag.boolean(
    True, 'Whether to used wheel to select sequences', is_key=None)
  sub_seq_len = Flag.integer(
    None, 'Length of sub-sequence used in seq_set.get_round_len or '
          'gen_rnn_batches', is_key=None)

  @property
  def sample_among_sequences(self):
    if self.sub_seq_len in [None, 0]: return False
    assert isinstance(self.sub_seq_len, int) and self.sub_seq_len > 0
    return True



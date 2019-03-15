from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class DataConfigs(object):
  train_size = Flag.integer(0, 'Training size')
  val_size = Flag.integer(0, 'Validation size')
  test_size = Flag.integer(0, 'Test size')
  sequence_length = Flag.integer(0, 'Sequence length', is_key=None)
  fixed_length = Flag.boolean(True, 'Whether to fix sequence length.'
                                    'used in AP, etc', is_key=None)
  cheat = Flag.boolean(True, '...', is_key=None)
  multiple = Flag.integer(1, '...', is_key=None)
  noisy = Flag.boolean(None, 'Whether XXX is noisy.', is_key=None)
  prediction_threshold = Flag.float(
    None, 'The prediction threshold used as an parameter for metric function',
    is_key=None)
  permuted = Flag.boolean(
    False, 'Whether to permute data (e.g. pMNIST)', is_key=None)

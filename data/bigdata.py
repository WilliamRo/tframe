from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe.data.base_classes import TFRData


class BigData(TFRData):
  """BigData manages .tfd files in a specific folder."""

  def __init__(self, data_dir):
    pass

  # region : Properties

  @property
  def size(self):
    return None

  @property
  def is_regular_array(self):
    return False

  # endregion : Properties

  # region : Public Methods

  def get_round_length(self, batch_size, num_steps=None):
    return None

  def gen_batches(self, batch_size, shuffle=False):
    return None

  def gen_rnn_batches(self, batch_size=1, num_steps=-1, shuffle=False):
    return None

  # endregion : Public Methods


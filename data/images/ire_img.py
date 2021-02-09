from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe.data.dataset import DataSet


class IrregularImageSet(DataSet):
  """An IrregularImageSet stores irregular images"""

  EXTENSION = 'tfdir'

  def __init__(self, features=None, targets=None, data_dict=None,
               name='irrset', **kwargs):
    # Call parent's constructor
    super().__init__(features, targets, data_dict, name, **kwargs)

  # region : Properties

  @DataSet.features.setter
  def features(self, val):
    if val is not None:
      if isinstance(val, (list, np.ndarray)):
        self.data_dict[self.FEATURES] = val
      else:
        raise TypeError('!! Unsupported feature type {}'.format(type(val)))

  @property
  def representative(self):
    data = list(self.data_dict.values())[0]
    assert isinstance(data, (np.ndarray, list))
    assert not self.is_rnn_input
    return data

  # endregion : Properties

  # region : Private Methods

  def _check_data(self):
    # Check data_dict
    if not isinstance(self.data_dict, dict) or len(self.data_dict) == 0:
      raise TypeError('!! data_dict must be a non-empty dictionary')

    valid_length = len(self.features)

    # Check each item in data_dict
    for name, data in self.data_dict.items():
      # Check type and length
      assert isinstance(data, (list, np.ndarray))
      # Check type and length
      if len(data) != valid_length: raise ValueError(
        '!! {} should be a list or np.ndarray with length {}'.format(
          name, valid_length))

  # endregion : Private Methods

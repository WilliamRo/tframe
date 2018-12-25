from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import checker
import numpy as np


class Statistic(object):
  def __init__(self, max_length=None):
    if max_length is not None: checker.check_positive_integer(max_length)
    self._max_length = max_length
    self._scalar_list = []
    self._scalar_count = 0
    self._accumulator = 0

  @property
  def last_value(self):
    if self._scalar_list: return self._scalar_list[-1]
    else: return None

  @property
  def average(self):
    return 1.0 * self._accumulator / self._scalar_count

  @property
  def running_average(self):
    return np.average(self._scalar_list)

  def record(self, scalar):
    assert np.isscalar(scalar)
    # Take down the new coming scalar
    self._scalar_list.append(scalar)
    if (self._max_length is not None and
        len(self._scalar_list) > self._max_length):
      self._scalar_list.pop(0)
    # Update global statistic
    self._scalar_count += 1
    self._accumulator += scalar

  def set_max_length(self, val):
    checker.check_positive_integer(val)
    self._max_length = val
    while len(self._scalar_list) > self._max_length: self._scalar_list.pop(0)


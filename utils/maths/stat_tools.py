from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import checker
import numpy as np


class Statistic(object):
  def __init__(self, max_length=None, keep_acc=True, keep_abs_acc=False,
               reduce_1st_dim=False):
    if max_length is not None: checker.check_positive_integer(max_length)
    self._max_length = max_length
    self._keep_acc = checker.check_type(keep_acc, bool)
    self._keep_abs_acc = checker.check_type(keep_abs_acc, bool)
    self._value_list = []
    self._value_count = 0
    self._accumulator = 0
    self._abs_accumulator = 0
    self._reduce_1st_dim = checker.check_type(reduce_1st_dim, bool)

  @property
  def last_value(self):
    if self._value_list: return self._value_list[-1]
    else: return None

  @property
  def average(self):
    assert self._keep_acc
    return 1.0 * self._accumulator / self._value_count

  @property
  def abs_average(self):
    assert self._keep_abs_acc
    return 1.0 * self._abs_accumulator / self._value_count

  @property
  def running_average(self):
    values = self._value_list
    if self._reduce_1st_dim: values = np.concatenate(values)
    return np.average(values, axis=0)

  @property
  def running_abs_average(self):
    values = self._value_list
    if self._reduce_1st_dim: values = np.concatenate(values)
    return np.average(np.abs(values), axis=0)

  def record(self, value):
    # Check type
    assert np.isscalar(value) or isinstance(value, np.ndarray)
    # Take down the new coming scalar
    self._value_list.append(value)
    if (self._max_length is not None and
        len(self._value_list) > self._max_length):
      self._value_list.pop(0)
    # Update global statistic
    # (values with shape [batch_size, dim] should be treated carefully)
    if not self._reduce_1st_dim:
      self._value_count += 1
      if self._keep_acc: self._accumulator += value
      if self._keep_abs_acc: self._abs_accumulator += np.abs(value)
    else:
      self._value_count += len(value)
      if self._keep_acc: self._accumulator += np.sum(value, axis=0)
      if self._keep_abs_acc: self._accumulator += np.sum(np.abs(value), axis=0)

  def set_max_length(self, val):
    checker.check_positive_integer(val)
    self._max_length = val
    while len(self._value_list) > self._max_length: self._value_list.pop(0)


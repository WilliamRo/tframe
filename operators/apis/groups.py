from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import inspect
from tframe import tf

from tframe import linker


class Groups(object):
  """This API should be distinguished from Group in core.group"""

  def __init__(self, config_string):
    """
    :param config_string: a string of format S1xN1+S2xN2+ ...
                          e.g. 4x25 represents 25 groups of size 4
    """
    self._groups = self._parse(config_string)

  # region : Configs

  @property
  def group_string(self):
    return '+'.join(['x'.join([str(n) for n in g]) for g in self._groups])

  @property
  def group_sizes(self): return [s * n for s, n in self._groups]

  @property
  def group_duplicates(self): return [n for _, n in self._groups]

  @property
  def total_size(self): return sum(self.group_sizes)

  @property
  def num_groups(self):
    return sum([n for _, n in self._groups])

  @staticmethod
  def _parse(config_string):
    assert isinstance(config_string, str)
    assert re.fullmatch(r'\d+x\d+(\+\d+x\d)*', config_string) is not None
    groups = []
    for gs in config_string.split('+'):
      assert isinstance(gs, str) and len(gs) > 2
      groups.append([int(n) for n in gs.split('x')])
    groups = tuple(sorted(groups, key=lambda g: g[0]))
    # Check groups (`g[0] must be unique` for g in groups)
    assert len(set([g[0] for g in groups])) == len(groups)
    for g in groups: assert g[0] > 0 and g[1] > 0
    return groups

  # endregion : Configs

  # region : Operations V1

  def _operate_over_groups(
      self, tensor, operator, sizes=None, reshape1=None, reshape2=None):
    assert isinstance(tensor, tf.Tensor) and callable(operator)
    if sizes is None: sizes = self.group_sizes
    # Split tensor
    splitted = linker.split_by_sizes(tensor, sizes)
    output_list = []
    for (s, n), data in zip(self._groups, splitted):
      dim1 = reshape1(s, n) if callable(reshape1) else s
      if n > 1: data = tf.reshape(data, [-1, dim1])
      data = operator(data)
      dim2 = reshape2(s, n) if callable(reshape2) else s * n
      if n > 1: data = tf.reshape(data, [-1, dim2])
      output_list.append(data)
    # Concatenate and return
    return linker.concatenate(output_list)

  def _binary_operate_over_groups(
      self, tensor1, tensor2, operator, sizes1=None, sizes2=None,
      reshape1_1=None, reshape1_2=None, reshape2=None):
    # Sanity check
    assert isinstance(tensor1, tf.Tensor) and isinstance(tensor2, tf.Tensor)
    if sizes1 is None: sizes1 = self.group_sizes
    if sizes2 is None: sizes2 = self.group_sizes
    # Split tensors
    splitted1 = linker.split_by_sizes(tensor1, sizes1)
    splitted2 = linker.split_by_sizes(tensor2, sizes2)
    output_list = []
    for (s, n), data1, data2 in zip(self._groups, splitted1, splitted2):
      # Reshape if necessary
      dim1_1 = reshape1_1(s, n) if callable(reshape1_1) else s
      dim1_2 = reshape1_2(s, n) if callable(reshape1_2) else s
      if n > 1:
        data1 = tf.reshape(data1, [-1, dim1_1])
        data2 = tf.reshape(data2, [-1, dim1_2])
      # Call operator
      num_args = len(inspect.getfullargspec(operator).args)
      if num_args == 2: args = []
      elif num_args == 3: args = [s * n]
      else: raise AssertionError(
        '!! Illegal operator with {} args'.format(num_args))
      data = operator(data1, data2, *args)
      # Reshape back if necessary
      dim2 = reshape2(s, n) if callable(reshape2) else s * n
      if n > 1: data = tf.reshape(data, [-1, dim2])
      # Add result to output list
      output_list.append(data)
    # Concatenate and return
    return linker.concatenate(output_list)

  def _softmax_over_groups(self, tensor):
    """The 'softmax over groups' activation implemented using tf.reshape"""
    operator = lambda t: tf.nn.softmax(t, axis=1)
    return self._operate_over_groups(tensor, operator)

  # endregion : Operations V1


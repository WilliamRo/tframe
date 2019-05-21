from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections

import tensorflow as tf

import tframe as tfr

from .maths.stat_tools import Statistic


class Monitor(object):

  def __init__(self):
    # independent value (which can be fetched without feed_dict) dictionary
    self._ind_val_dict = collections.OrderedDict()
    self._weights_list = []
    self._grad_ops = []

  @property
  def grad_ops_list(self): return self._grad_ops

  # region : Public Methods

  def register_weights(self, weights):
    if not isinstance(weights, (tuple, list)): weights = [weights]
    tfr.checker.check_fetchable(weights)
    for w in weights:
      assert w not in self._weights_list
      self._weights_list.append(w)
      self._ind_val_dict[w] = Statistic(max_length=10, keep_acc=False)

  def register_loss(self, loss):
    """Currently tensors inside while_loop are not considered"""
    assert isinstance(loss, tf.Tensor)
    self._grad_ops = tf.gradients(loss, self._weights_list)

  def record(self, grads):
    assert isinstance(grads, list) and len(grads) == len(self._weights_list)
    for w, g in zip(self._weights_list, grads):
      s = self._ind_val_dict[w]
      assert isinstance(s, Statistic)
      s.record(g)

  def update_dict(self, tensor_dict):
    """This method will only be called by trainer._get_variable_to_export"""
    assert isinstance(tensor_dict, dict)
    for w in self._weights_list:
      assert isinstance(w, (tf.Tensor, tf.Variable))
      key = '/'.join(w.name.split('/')[1:])
      key = 'grad({})'.format(key)
      s = self._ind_val_dict[w]
      assert isinstance(s, Statistic)
      tensor_dict[key] = s.running_abs_average

  # endregion : Public Methods

  # region : Private Methods

  # endregion : Private Methods

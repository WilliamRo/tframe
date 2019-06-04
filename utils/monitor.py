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

    self._researchers = []

  @property
  def grad_ops_list(self): return self._grad_ops

  @property
  def grad_dict(self):
    """This property will only be used by trainer._get_variable_to_export"""
    if not tfr.hub.export_weight_grads: return

    grads = collections.OrderedDict()
    for w in self._weights_list:
      assert isinstance(w, (tf.Tensor, tf.Variable))
      key = '/'.join(w.name.split('/')[1:])
      key = 'grad({})'.format(key)
      s = self._ind_val_dict[w]
      assert isinstance(s, Statistic)
      grads['|{}|'.format(key)] = s.running_abs_average
      grads[key] = s.running_average
      # grads['|{}|'.format(key)] = s.abs_average
    # Let researchers do their job
    for r in self._researchers: r(grads)

    return grads

  # region : Public Methods

  def register_weights(self, weights):
    """This method will only be used in Net.variable_extractor which will
       is called at the end of Net/RNet's _link method.
       Weight vars (with `w` in their name) in Net's var_list will be registered
       to monitor using this api.
    """
    if not isinstance(weights, (tuple, list)): weights = [weights]
    tfr.checker.check_fetchable(weights)
    for w in weights:
      assert w not in self._weights_list
      self._weights_list.append(w)
      self._ind_val_dict[w] = Statistic(max_length=20, keep_abs_acc=True)

  def register_loss(self, loss):
    """Currently tensors inside while_loop are not considered.
       Called during predictor._building
    """
    assert isinstance(loss, tf.Tensor)
    self._grad_ops = tf.gradients(loss, self._weights_list)

  def record(self, grads):
    """This method will be only called in train.update_model.
       Gradient statistics will be recorded.
    """
    assert isinstance(grads, list) and len(grads) == len(self._weights_list)
    for w, g in zip(self._weights_list, grads):
      s = self._ind_val_dict[w]
      assert isinstance(s, Statistic)
      s.record(g)

  def register_researcher(self, researcher):
    """A researcher takes a tensor_dict to export as input"""
    assert callable(researcher)
    self._researchers.append(researcher)

  def get_weight_stats(self, weights):
    return self._ind_val_dict[weights]

  # endregion : Public Methods

  # region : Private Methods

  # endregion : Private Methods

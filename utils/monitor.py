from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections

from tframe import tf

import tframe as tfr

from .maths.stat_tools import Statistic


class Monitor(object):
  """Monitor is originally developed for monitoring gradients.
     For monitoring neuron activations, an activation filter must be registered.
     An activation filter takes a tf.Tensor as input and returns
     0: ignore this tensor
     1: register as Type I (into _tensor_stats_dict)
     2: register as Type II (into context.variables_to_export)
  """

  def __init__(self):
    # Attributes for monitoring general tensors
    self._activation_filter = None
    self._tensor_stats_dict = collections.OrderedDict()

    # :: Attributes specifically for monitoring weights
    # independent value (which can be fetched without feed_dict) dictionary
    self._weights_list = []  # TODO: consider remove this attribute
    self._grad_ops = []
    self._weight_grad_dict = collections.OrderedDict()
    self._grad_researchers = []

    # TODO: beta
    self._weight_history = collections.OrderedDict()
    self._weight_flip_count = collections.OrderedDict()

  # region : Properties

  # region : Properties for general tensors

  @property
  def tensor_fetches(self):
    fetches = list(self._tensor_stats_dict.keys())
    # if len(fetches) == 0:
    #   raise AssertionError('No general tensor fetches found in monitor')
    return fetches

  @property
  def stats_dict(self):
    """This property should onl be used in Trainer._get_variables_to_export
       method for exporting general stats to note. grads are regarded as
       special stats with separate logic.
    """
    stats = collections.OrderedDict()
    for tensor, stat in self._tensor_stats_dict.items():
      assert isinstance(tensor, tf.Tensor) and isinstance(stat, Statistic)
      key = '/'.join(tensor.name.split('/')[1:3])
      # stats[key] = stat.running_average  # only abs_avg is necessary currently
      stats['|{}|'.format(key)] = stat.running_abs_average
    return stats

  # endregion : Properties for general tensors

  # region : Properties for weights grads

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
      s = self._weight_grad_dict[w]
      assert isinstance(s, Statistic)
      grads['|{}|'.format(key)] = s.running_abs_average
      grads[key] = s.running_average
      # grads['|{}|'.format(key)] = s.abs_average
    # Let researchers do their job
    for r in self._grad_researchers: r(grads)

    return grads

  # endregion : Properties for weights grads

  # endregion : Properties

  # region : Public Methods

  # region : Methods for monitoring general tensors

  def register_activation_filter(self, act_filter):
    """There should be only 1 activation filter."""
    assert self._activation_filter is None
    assert callable(act_filter)
    self._activation_filter = act_filter

  def register_tensor(
      self, tensor, name='tensor',
      reduce_1st_dim=False, keep_acc=False, keep_abs_acc=False):
    if not callable(self._activation_filter):
      raise ValueError(
        'context.monitor.register_activation filter should be called before'
        ' building model when options like th.export_activations are set to '
        'True')
    assert isinstance(tensor, tf.Tensor)
    # Determine to ignore or register tensor into the corresponding dict
    if self._activation_filter(tensor) in (False, '0', 0): return
    if self._activation_filter(tensor) in (1, '1'):
      assert tensor not in self._tensor_stats_dict
      self._tensor_stats_dict[tensor] = Statistic(
        max_length=tfr.hub.stats_max_length, keep_acc=keep_acc,
        keep_abs_acc=keep_abs_acc, reduce_1st_dim=reduce_1st_dim)
    else: tfr.context.add_tensor_to_export(name, tensor)

  def register_stats(self, tensors):
    """stats will be exported to note in Trainer._get_variables_to_export
       method
    """
    if not isinstance(tensors, (tuple, list)): tensors = [tensors]
    tfr.checker.check_fetchable(tensors)
    for t in tensors:
      assert t not in self._tensor_stats_dict
      self._weight_grad_dict[t] = Statistic(
        max_length=tfr.hub.stats_max_length, keep_abs_acc=True)

  def record_tensors(self, tensors):
    """Record customized tensors such as activations"""
    if not isinstance(tensors, (tuple, list)): tensors = [tensors]
    assert len(tensors) == len(self._tensor_stats_dict)
    for t, s in zip(tensors, self._tensor_stats_dict.values()):
      assert isinstance(s, Statistic)
      s.record(t)

  # endregion : Methods for monitoring general tensors

  # region : Methods for monitoring weight gradients

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
      self._weight_grad_dict[w] = Statistic(
        max_length=tfr.hub.stats_max_length, keep_abs_acc=True)
      # TODO: BETA
      if tfr.hub.monitor_weight_history:
        self._weight_history[w] = Statistic(max_length=2, keep_abs_acc=False)

  def register_loss(self, loss):
    """Currently tensors inside while_loop are not considered.
       Called during predictor._building
    """
    assert isinstance(loss, tf.Tensor)
    self._grad_ops = tf.gradients(loss, self._weights_list)

  def record_grads(self, grads):
    """This method will be only called in train.update_model.
       Gradient statistics will be recorded.
    """
    assert isinstance(grads, list) and len(grads) == len(self._weights_list)
    for w, g in zip(self._weights_list, grads):
      s = self._weight_grad_dict[w]
      assert isinstance(s, Statistic)
      s.record(g)

  def register_grad_researcher(self, researcher):
    """A researcher takes a tensor_dict to export as input.
       Researchers are usually registered in XX_mu.py module.
    """
    assert callable(researcher)
    self._grad_researchers.append(researcher)

  def get_weight_stats(self, weights):
    return self._weight_grad_dict[weights]

  # endregion : Methods for monitoring weight gradients

  # region : Methods for monitoring weight s

  def record_weights(self):
    """This method will be only called in train.update_model.
       Weights statistics will be recorded.
    """
    weights = tfr.context.trainer.model.agent.session.run(self._weights_list)
    for w, current_w in zip(self._weights_list, weights):
      s = self._weight_history[w]
      assert isinstance(s, Statistic)
      last_w = s.last_value
      s.record(current_w)
      if last_w is None: continue

      if not tfr.hub.monitor_weight_flips: continue

      # Calculate flip matrix and update flip matrices
      flips = current_w * last_w < 0
      alpha, beta = tfr.hub.flip_alpha, tfr.hub.flip_beta
      assert 0 <= alpha <=1 and 0 <= beta <=1
      self._weight_flip_count[w] = (
          beta * self.get_weight_flip_count(w) + alpha * flips)

  def get_weight_flip_count(self, weights):
    assert weights in self._weight_history
    if weights not in self._weight_flip_count:
      count = np.zeros(shape=weights.shape.as_list(), dtype=int)
      self._weight_flip_count[weights] = count
    return self._weight_flip_count[weights]

  # endregion : Methods for monitoring weights

  # endregion : Public Methods

  # region : Private Methods

  # endregion : Private Methods

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import pedia
from tframe.core import with_graph

from tframe import Predictor
from tframe.nets.net import Net
from tframe.models import Feedforward

from tframe import losses
from tframe import metrics


class Bamboo(Predictor):
  def __init__(self, mark=None):
    # Call parent's initializer
    Predictor.__init__(self, mark)
    # Private fields
    self._branch_index = -1
    self._output_list = []
    self._losses = []
    self._train_ops = []
    self._metrics = []

  # region : Build

  @with_graph
  def _build(self, loss='cross_entropy', optimizer=None,
             metric=None, metric_is_like_loss=True, metric_name='Metric'):
    Feedforward._build(self)
    # Check branch shapes
    output_shape = self.outputs.shape_list
    for b_out in self.branch_outputs:
      assert isinstance(b_out, tf.Tensor)
      if b_out.get_shape().as_list() != output_shape:
        raise ValueError('!! Branch outputs in bamboo should have the same'
                         ' shape as the trunk output')
    # Initiate targets and add it to collection
    self._plug_target_in(output_shape)

    # Generate output list
    self._output_list = self.branch_outputs + [self.outputs.tensor]

    # Define losses
    loss_function = losses.get(loss)
    with tf.name_scope('Loss'):
      # Add branch outputs
      for output in self._output_list:
        assert isinstance(output, tf.Tensor)
        self._losses.append(loss_function(self._targets.tensor, output))

    # Define metrics
    metric_function = metrics.get(metric)
    if metric_function is not None:
      with tf.name_scope('Metric'):
        for output in self._output_list:
          self._metrics.append(metric_function(self._targets.tensor, output))
        self.key_metric.plug(
          self._metrics[-1], as_loss=metric_is_like_loss, symbol=metric_name)

    # Define train step
    self._define_train_step(optimizer)

    # Set default branch
    self.set_branch_index(-1)

    # Sanity check
    assert len(self._losses) == len(self._metrics) == len(
      self.branch_outputs) + 1

  @with_graph
  def _define_train_step(self, optimizer=None, var_list=None):
    assert len(self._losses) > 0
    with tf.name_scope('Optimizer'):
      if optimizer is None: optimizer = tf.train.AdamOptimizer(1e-4)
      self._optimizer = optimizer
      loss_index = 0
      var_list = []
      for i, net in enumerate(self.children):
        assert isinstance(net, Net)
        var_list += net.var_list
        if net.is_branch or i == len(self.children) - 1:
          self._train_ops.append(optimizer.minimize(
            loss=self._losses[loss_index], var_list=var_list))
          loss_index += 1
          var_list = []

    assert len(self._losses) == len(self._train_ops)

  # endregion : Build

  # region : Train

  @with_graph
  def train(self, *args, branch_index=0, **kwargs):
    self.set_branch_index(branch_index)
    # TODO
    freeze = kwargs.get('freeze', True)
    if not freeze:
      self.train_step.substitute(self._optimizer.minimize(self.loss.tensor))
    # Call parent's train method
    Predictor.train(self, *args, **kwargs)

  def grow(self):
    pass

  # endregion : Train

  # region : Private Methods

  def _clone_branch(self, index):
    """Branch clone needs two branches have the same structure"""
    branches = [net for net in self.children if net.is_branch]
    branches += self.children[-1]
    assert index < len(branches)

  # endregion : Private Methods

  # region : Public Methods

  def set_branch_index(self, index):
    # Sanity check
    if not 0 <= index < len(self._losses) and index != -1:
      raise IndexError('!! branch index should be between {} and {}'.format(
        0, len(self._losses)))

    self._branch_index = index
    self.outputs.substitute(self._output_list[index])
    self.loss.substitute(self._losses[index])
    self.train_step.substitute(self._train_ops[index])
    self.key_metric.substitute(self._metrics[index])

  def predict(self, data, **kwargs):
    index = kwargs.get('branch_index', 0)
    self.set_branch_index(index)
    # Call parent's predict method
    return Predictor.predict(self, data, **kwargs)

  # endregion : Public Methods







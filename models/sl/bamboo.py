from __future__ import absolute_import
from __future__ import division
from __future__ import division

import numpy as np
import tensorflow as tf

from tframe import pedia
from tframe.core import with_graph

from tframe import Predictor
from tframe.nets.net import Net
from tframe.models import Feedforward

from tframe import console
from tframe import losses
from tframe import metrics
from tframe import TFData

from tframe import hub


class Bamboo(Predictor):
  def __init__(self, mark=None):
    # Call parent's initializer
    Predictor.__init__(self, mark)
    # Private fields
    self._output_list = []
    self._losses = []
    self._metrics = []
    self._train_ops = []
    self._branch_index = 0


  def set_branch_index(self, index):
    # Sanity check
    if not 0 <= index < len(self._losses):
      raise IndexError('!! branch index should be between {} and {}'.format(
        0, len(self._losses)))

    self._branch_index = index
    self._loss = self._losses[index]
    self._metric = self._metrics[index]
    self._train_step = self._train_ops[index]
    self._outputs = self._output_list[index]


  @with_graph
  def build(self, loss='cross_entropy', optimizer=None,
            metric=None, metric_name='Metric'):
    Feedforward.build(self)
    # Check branch shapes
    output_shape = self._outputs.get_shape().as_list()
    for b_out in self.branch_outputs:
      assert isinstance(b_out, tf.Tensor)
      if b_out.get_shape().as_list() != output_shape:
        raise ValueError('!! Branch outputs in bamboo should have the same'
                         ' shape as the trunk output')
    # Initiate targets and add it to collection
    self._targets = tf.placeholder(self._outputs.dtype, output_shape,
                                   name='targets')
    tf.add_to_collection(pedia.default_feed_dict, self._targets)

    # Generate output list
    output_list = self.branch_outputs + [self._outputs]

    # Define losses
    loss_function = losses.get(loss)
    with tf.name_scope('Loss'):
      # Add branch outputs
      for output in output_list:
        self._losses.append(loss_function(self._targets, output))

    # Define metrics
    metric_function = metrics.get(metric)
    if metric_function is not None:
      pedia.memo[pedia.metric_name] = metric_name
      with tf.name_scope('Metric'):
        for output in output_list:
          self._metrics.append(metric_function(self._targets, output))

    # Define train step
    self._define_train_step(optimizer)

    # Sanity check
    assert len(self._losses) == len(self._metrics) == len(
      self.branch_outputs) + 1

    # Print status and model structure
    self.show_building_info(FeedforwardNet=self)

    # Launch session
    self.launch_model(hub.overwrite)

    # Set built flag
    self._output_list = output_list
    self._built = True


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


  @with_graph
  def train(self, *args, branch_index=0, **kwargs):
    self.set_branch_index(branch_index)
    # TODO
    freeze = kwargs.get('freeze', True)
    if not freeze: self._train_step = self._optimizer.minimize(self._loss)
    # Call parent's train method
    Predictor.train(self, *args, **kwargs)


  def predict(self, data, **kwargs):
    index = kwargs.get('branch_index', 0)
    self.set_branch_index(index)
    # Call parent's predict method
    return Predictor.predict(self, data, **kwargs)







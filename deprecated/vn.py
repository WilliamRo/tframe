from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

from tframe.models.model import Model
from tframe.nets.net import Net

from tframe import console

from tframe import pedia
from tframe import losses
from tframe import metrics
from tframe import DataSet
from tframe.core import with_graph

from tframe.layers.layer import Layer
from tframe.layers import Input
from tframe.layers.homogeneous import Homogeneous


class VolterraNet(Model):
  """ A class for Volterra Networks"""

  def __init__(self, degree, depth, mark=None, max_volterra_order=3, **kwargs):
    # Check parameters
    if degree < 1: raise ValueError('!! Degree must be a positive integer')
    if depth < 0: raise ValueError('!! Depth must be a positive integer')

    # Call parent's constructor
    Model.__init__(self, mark)

    # Initialize fields
    self.degree = degree
    self.depth = depth
    self._max_volterra_order = min(max_volterra_order, degree)
    self.T = {}
    self._input = Input([depth], name='input')
    self._output = None
    self._target = None
    self._alpha = 1.1
    self._outputs = {}

    # Initialize operators in each degree
    orders = kwargs.get('orders', None)
    if orders is None: orders = list(range(1, self.degree + 1))
    self.orders = orders
    self._init_T()

  # region : Properties

  @property
  def linear_coefs(self):
    if 1 not in self.orders: return None
    coefs = self._session.run(self.T[1].chain[0].chain[0].weights)
    return coefs.flatten()

  @property
  def operators(self):
    od = collections.OrderedDict()
    for i in range(1, self.degree + 1):
      if i not in self.orders: continue
      od[i] = self.T[i]
    return od

  @property
  def description(self):
    result = ''
    for key, val in self.operators.items():
      assert isinstance(val, Net)
      result += 'T[{}]: {}\n'.format(key, val.structure_string())
    return result

  # endregion : Properties

  # region : Building

  @with_graph
  def _build(self, loss='euclid', optimizer=None, homo_strength=1.0,
             metric=None, metric_name='Metric'):
    """Build model"""
    # Set summary place holder
    default_summaries = []
    print_summaries = []
    # Define output
    for order, op in self.T.items(): self._outputs[order] = op()
    with tf.name_scope('Outputs'):
      self._output = tf.add_n(list(self._outputs.values()), name='output')

    self._target = tf.placeholder(
      self._output.dtype, self._output.get_shape(), name='target')
    tf.add_to_collection(pedia.default_feed_dict, self._target)

    # Define loss
    loss_function = losses.get(loss)
    with tf.name_scope('Loss'):
      # All losses in loss list will be added
      loss_list = []

      # Delta loss
      with tf.name_scope('Delta'):
        delta_loss = loss_function(self._target, self._output)
        loss_list.append(delta_loss)
        default_summaries.append(
          tf.summary.scalar('delta_loss_sum', delta_loss))

      # Homogeneous loss
      with tf.name_scope('Homogeneous'):
        homo_list = []
        # Calculate homo-loss for each order
        for order, op in self.T.items():
          if order in range(1, self._max_volterra_order + 1): continue
          coef = self._alpha ** order
          truth_k = self._outputs[order] * coef
          pred_k = op(self._input.place_holder * self._alpha)

          # Calculate loss
          homo_loss_k = tf.norm(truth_k - pred_k,
                                name='home_loss_{}'.format(order))
          # homo_list.append(numerator / coef)
          homo_list.append(homo_loss_k)

          # Add summary
          default_summaries.append(tf.summary.scalar(
            'homo_loss_{}_sum'.format(order), homo_loss_k))

        # Add all homogeneous losses
        if len(homo_list) > 0:
          homo_loss = tf.add_n(homo_list, 'homo_loss') * homo_strength
          loss_list.append(homo_loss)

      # Try to add regularization loss
      reg_list = [op.regularization_loss for op in self.T.values()
                  if op.regularization_loss is not None]
      if len(reg_list) > 0:
        with tf.name_scope('WeightNorm'):
          weight_norm = tf.add_n(reg_list, name='reg_loss')
          loss_list.append(weight_norm)
          # tf.summary.scalar('reg_loss_sum', weight_norm)

      # Add all losses
      self._loss = tf.add_n(loss_list, name='loss')
      # tf.summary.scalar('total_loss', self._loss)

    # Define metric
    metric_function = metrics.get(metric)
    if metric_function is not None:
      pedia.memo[pedia.metric_name] = metric_name
      with tf.name_scope('Metric'):
        self._metric = metric_function(self._target, self._output)
        print_summaries.append(tf.summary.scalar('metric_sum', self._metric))

    # Merge summaries
    self._merged_summary = tf.summary.merge(default_summaries,
                                            name='default_summaries')
    if print_summaries is not None:
      self._print_summary = tf.summary.merge(print_summaries)


    # Define train step
    self._define_train_step(optimizer)

    # Print status and model structure
    self._show_building_info(
      **{'T[{}]'.format(key): val for key, val in self.operators.items()})

    # Launch session
    self.launch_model(FLAGS.overwrite and FLAGS.train)

    # Set built flag
    self._built = True

  # endregion : Building

  # region : Private Methods

  def _init_T(self):
    # Add empty nets to each degree
    for n in self.orders:
      self.T[n] = Net('T{}'.format(n))
      self.T[n].add(self._input)

    # Initialize volterra part
    for order in range(1, self._max_volterra_order + 1):
      self.T[order].add(Homogeneous(order))

  # endregion : Private Methods

  # region : Public Methods

  def volterra_coefs(self, orders=None):
    if orders is None: orders = self.orders
    od = collections.OrderedDict()
    return od


  def add(self, order, layer):
    if not order in self.orders:
      raise ValueError('!! Model does not have order {}'.format(order))
    if not isinstance(layer, Layer):
      raise ValueError('!! The second parameter should be an instance of Layer')
    if order in range(1, self._max_volterra_order + 1):
      raise ValueError('!! Order 1 to {} are fixed'.format(
        self._max_volterra_order))

    self.T[order].add(layer)


  # TODO: Exactly the same as predict method in predictor.py
  def predict(self, data):
    # Sanity check
    if not isinstance(data, DataSet):
      raise TypeError('!! Input data must be an instance of TFData')
    if not self.built: raise ValueError('!! Model not built yet')
    if self._session is None:
      self.launch_model(overwrite=False)

    if data.targets is None:
      outputs = self._session.run(
        self._output,
        feed_dict=self._get_default_feed_dict(data, is_training=False))
      return outputs
    else:
      outputs, loss = self._session.run(
        [self._output, self._loss],
        feed_dict=self._get_default_feed_dict(data, is_training=False))
      return outputs, loss

  # endregion : Public Methods

  """For some reason, do not delete this line"""



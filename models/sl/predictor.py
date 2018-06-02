from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe.models.model import Model
from tframe.models.feedforward import Feedforward
from tframe.models.recurrent import Recurrent

from tframe import console
from tframe import losses
from tframe import pedia
from tframe import metrics
from tframe import DataSet

from tframe import hub
from tframe import InputTypes
from tframe.core import with_graph
from tframe.core import TensorSlot

from tframe.trainers import TrainerHub
from tframe.data.base_classes import TFRData


class Predictor(Feedforward, Recurrent):
  """A feedforward or a recurrent predictor"""
  model_name = 'Predictor'

  def __init__(self, mark=None, net_type=Feedforward):
    """
    Construct a Predictor
    :param mark: model mark
    :param net_type: \in {Feedforward, Recurrent}
    """
    if not net_type in (Feedforward, Recurrent):
      raise TypeError('!! Unknown net type')
    self.master = net_type
    # Call parent's constructor
    net_type.__init__(self, mark)
    # Attributes
    self._targets = TensorSlot(self, 'targets')

  # region : Properties

  @property
  def description(self):
    return '{}: {}'.format(self.master.__name__, self.structure_string())

  @property
  def input_type(self):
    if self.master is Feedforward: return InputTypes.BATCH
    else: return InputTypes.RNN_BATCH

  # endregion : Properties

  # region : Build

  @with_graph
  def build_as_regressor(
      self, optimizer=None, loss='euclid',
      metric='rms_ratio', metric_is_like_loss=True, metric_name='Err %'):
    self.build(
      optimizer=optimizer, loss=loss, metric=metric, metric_name=metric_name,
      metric_is_like_loss=metric_is_like_loss)

  @with_graph
  def build(self, optimizer=None, loss='euclid', metric=None,
            metric_is_like_loss=True, metric_name='Metric', **kwargs):
    Model.build(
      self, optimizer=optimizer, loss=loss, metric=metric,
      metric_name=metric_name, metric_is_like_loss=metric_is_like_loss)

  def _build(self, optimizer=None, loss='euclid',
             metric=None, metric_is_like_loss=True, metric_name='Metric',
             **kwargs):
    # Call parent's build method
    # Usually output tensor has been plugged into Model._outputs slot
    self.master._build(self)
    assert self.outputs.activated

    # Initiate targets and add it to collection
    self._plug_target_in(self.outputs.shape_list)

    # Define loss
    loss_function = losses.get(loss)
    with tf.name_scope('Loss'):
      loss_tensor = loss_function(self._targets.tensor, self.outputs.tensor)
      # TODO: with or without regularization loss?
      if hub.summary:
        tf.add_to_collection(pedia.train_step_summaries,
                             tf.summary.scalar('loss_sum', loss_tensor))
      # Try to add regularization loss
      reg_loss = self.regularization_loss
      if reg_loss is not None: loss_tensor += reg_loss
      # Plug in
      self.loss.plug(loss_tensor)

    # Define metric
    if metric is not None:
      metric_function = metrics.get(metric)
      with tf.name_scope('Metric'):
        metric_tensor = metric_function(
          self._targets.tensor, self._outputs.tensor)
        self._metric.plug(metric_tensor, as_loss=metric_is_like_loss,
                          symbol=metric_name)
        if hub.summary:
          tf.add_to_collection(
            pedia.validation_summaries,
            tf.summary.scalar('metric_sum', self._metric.tensor))

    # Merge summaries
    self._merge_summaries()

    # Define train step
    self._define_train_step(optimizer)

  def _plug_target_in(self, shape):
    target_tensor = tf.placeholder(hub.dtype, shape, name='targets')
    self._targets.plug(target_tensor, collection=pedia.default_feed_dict)

  # endregion : Build

  # region : Train

  def begin_round(self, **kwargs):
    if self.master is Recurrent:
      th = kwargs.get('th')
      assert isinstance(th, TrainerHub)
      self.reset_state(th.batch_size)

  def update_model(self, data_batch, **kwargs):
    if self.master is Feedforward:
      return Feedforward.update_model(self, data_batch, **kwargs)
    # Update recurrent model
    feed_dict = self._get_default_feed_dict(data_batch, is_training=True)
    results = self._update_group.run(feed_dict)
    self._state_array = results.pop(self._state)
    return results

  # endregion : Train

  # region : Public Methods

  def predict(self, data, batch_size=None, extractor=None, **kwargs):
    outputs = []
    for batch in self.get_data_batches(data, batch_size):
      batch = self._sanity_check_before_use(batch)
      feed_dict = self._get_default_feed_dict(batch, is_training=False)
      output = self._outputs.run(feed_dict)
      if extractor is not None: output = extractor(output)
      outputs.append(output)
    return np.concatenate(outputs)

  def evaluate_model(self, data, batch_size=None, **kwargs):
    # Check metric
    if not self.metric.activated: raise AssertionError('!! Metric not defined')
    # Show status
    console.show_status('Evaluating {} ...'.format(data.name))
    result = self.validate_model(data, batch_size, allow_sum=False)[self.metric]
    console.supplement('{} = {:.3f}'.format(self.metric.symbol, result))

  # endregion : Public Methods

  # region : Private Methods

  def _get_default_feed_dict(self, batch, is_training):
    feed_dict = Feedforward._get_default_feed_dict(self, batch, is_training)
    if self.master is Recurrent:
      batch_size = None if is_training else batch.size
      feed_dict.update(self._get_state_dict(batch_size=batch_size))
    return feed_dict

  # endregion : Private Methods








from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

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
  def _build(self, loss='cross_entropy', optimizer=None,
             metric=None, metric_is_like_loss=True, metric_name='Metric'):
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

  def predict(self, data, additional_fetches=None, **kwargs):
    # Sanity check
    self._sanity_check_before_use(data)

    fetches = [self._outputs]
    if additional_fetches is not None:
      fetches += list(additional_fetches)
    feed_dict = self._get_default_feed_dict(data, is_training=False)
    return self._outputs.run(fetches, feed_dict=feed_dict)

  def evaluate_model(self, data, **kwargs):
    # Sanity check
    self._sanity_check_before_use(data)

    # Check metric
    if self.metric is None:
      raise AssertionError('!! Failed to evaluate due to missing metric')

    # Show status
    console.show_status('Evaluating {}'.format(data.name))
    # Get result
    feed_dict = self._get_default_feed_dict(data, False)
    result = self.metric.fetch(feed_dict)
    console.supplement('{} = {:.3f}'.format(self.metric.symbol, result))

  # endregion : Public Methods

  # region : Private Methods

  def _get_default_feed_dict(self, batch, is_training):
    feed_dict = Feedforward._get_default_feed_dict(self, batch, is_training)
    if self.master is Recurrent:
      batch_size = None if is_training else 1
      feed_dict.update(self._get_state_dict(batch_size=batch_size))
    return feed_dict

  def _sanity_check_before_use(self, data):
    if not isinstance(data, DataSet):
      raise TypeError('!! Input data must be an instance of TFData')
    if not self.built: raise ValueError('!! Model not built yet')
    if not self.launched: self.launch_model(overwrite=False)

  # endregion : Private Methods








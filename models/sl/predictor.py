from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.models.feedforward import Feedforward
from tframe.models.recurrent import Recurrent

from tframe import losses
from tframe import pedia
from tframe import metrics
from tframe import TFData

from tframe import hub
from tframe import InputTypes
from tframe.core import with_graph
from tframe.core import TensorSlot


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
    self._default_net = self
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

  @property
  def metric_is_accuracy(self):
    return pedia.memo[pedia.metric_name] == pedia.Accuracy

  # endregion : Properties

  # region : Build

  @with_graph
  def _build(self, loss='cross_entropy', optimizer=None,
             metric=None, metric_is_like_loss=True, metric_name='Metric'):
    # Call parent's build method
    self.master._build(self)

    # Summary placeholder
    default_summaries = []
    validation_summaries = []

    # Initiate targets and add it to collection
    targets = tf.placeholder(
      self.outputs.dtype, self.outputs.shape_list, name='targets')
    self._targets.plug(targets, collection=pedia.default_feed_dict)

    # Define loss
    loss_function = losses.get(loss)
    with tf.name_scope('Loss'):
      loss_tensor = loss_function(self._targets.tensor, self.outputs.tensor)
      # TODO: with or without regularization loss?
      default_summaries.append(tf.summary.scalar('loss_sum', self._loss))
      # Try to add regularization loss
      reg_loss = self.regularization_loss
      if reg_loss is not None: loss_tensor += reg_loss
      # Plug in
      self.loss.plug(loss_tensor)

    # Define metric
    metric_function = metrics.get(metric)
    if metric_function is not None:
      with tf.name_scope('Metric'):
        metric_tensor = metric_function(
          self._targets.tensor, self._outputs.tensor)
        self._metric.plug(metric_tensor, as_loss=metric_is_like_loss,
                          symbol=metric_name)
        validation_summaries.append(
          tf.summary.scalar('metric_sum', self._metric.tensor))

    # Merge summaries
    merged_summary = tf.summary.merge(default_summaries)
    self._merged_summary.plug(merged_summary)
    if len(validation_summaries) > 0:
      self._validation_summary.plug(tf.summary.merge(validation_summaries))

    # Define train step
    self._define_train_step(optimizer)

  # endregion : Build

  # region : Public Methods

  def predict(self, data, additional_fetches=None, **kwargs):
    # Sanity check
    if not isinstance(data, TFData):
      raise TypeError('!! Input data must be an instance of TFData')
    if not self.built: raise ValueError('!! Model not built yet')
    if not self.launched: self.launch_model(overwrite=False)

    fetches = [self._outputs]
    if additional_fetches is not None:
      fetches += list(additional_fetches)
    feed_dict = self._get_default_feed_dict(data, is_training=False)
    return self._outputs.run(fetches, feed_dict=feed_dict)

  # endregion : Public Methods








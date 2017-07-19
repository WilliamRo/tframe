from __future__ import absolute_import

import tensorflow as tf

from .feedforward import Feedforward

from .. import losses
from .. import pedia
from .. import metrics

from .. import FLAGS


class Predictor(Feedforward):
  def __init__(self, mark=None):
    Feedforward.__init__(self, mark)
    self._targets = None

  def build(self, loss='cross_entropy', optimizer=None,
             metric=None, metric_name='Metric'):
    Feedforward.build(self)
    # Initiate targets and add it to collection
    self._targets = tf.placeholder(self.outputs.dtype, self.outputs.get_shape(),
                                   name='targets')
    tf.add_to_collection(pedia.default_feed_dict, self._targets)

    # Define loss
    loss_function = losses.get(loss)
    with tf.name_scope('Loss'):
      self._loss = loss_function(self._targets, self.outputs)
      tf.summary.scalar('loss_sum', self._loss)

    # Define metric
    metric_function = metrics.get(metric)
    if metric_function is not None:
      pedia.memo[pedia.metric_name] = metric_name
      with tf.name_scope('Metric'):
        self._metric = metric_function(self._targets, self.outputs)

    # Define train step
    self._define_train_step(optimizer)

    # Print status and model structure
    self.show_building_info(FeedforwardNet=self)

    # Launch session
    self.launch_model(FLAGS.overwrite)

  def predict(self, data):
    if self.outputs is None:
      raise ValueError('Model not built yet')

    if self._session is None:
      self.launch_model(overwrite=False)

    outputs, loss = self._session.run(
      [self.outputs, self._loss],
      feed_dict=self._get_default_feed_dict(data, train=False))

    return outputs, loss






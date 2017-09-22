from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .predictor import Predictor
from ..utils import console
from .. import config
from .. import pedia


class Classifier(Predictor):
  def __init__(self, mark=None):
    Predictor.__init__(self, mark)
    self._sum_train_acc = None
    self._sum_val_acc = None


  def build(self, loss='cross_entropy', optimizer=None):
    # TODO: ... do some compromise
    config.block_validation = True
    Predictor.build(self, loss, optimizer, metric='accuracy',
                    metric_name=pedia.Accuracy)

    self._sum_train_acc = tf.summary.scalar('train_acc', self._metric)
    self._sum_val_acc = tf.summary.scalar('val_acc', self._metric)


  def _update_model(self, data_batch, **kwargs):
    feed_dict = self._get_default_feed_dict(data_batch, is_training=True)

    summary, _ = self._session.run(
      [self._sum_train_acc, self._train_step], feed_dict=feed_dict)

    assert isinstance(self._summary_writer, tf.summary.FileWriter)
    self._summary_writer.add_summary(summary, self._counter)

    return {}


  def _print_progress(self, epc, start_time, info_dict, **kwargs):
    # Sanity check
    assert self._metric is not None
    if self._validation_set is None:
        raise ValueError('Validation set is None')

    # Reset dict
    info_dict = {}
    data_batch = kwargs.get('data_batch', None)
    assert data_batch is not None
    train_acc = self._metric.eval(self._get_default_feed_dict(
      data_batch, is_training=False))

    feed_dict = self._get_default_feed_dict(
      self._validation_set, is_training=False)
    summary, val_acc = self._session.run(
      [self._sum_val_acc, self._metric], feed_dict=feed_dict)
    self._summary_writer.add_summary(summary, self._counter)

    info_dict['Training Accuracy'] = train_acc
    info_dict['Validation Accuracy'] = val_acc

    # Call predictor's _print_progress
    Predictor._print_progress(self, epc, start_time, info_dict)


  def evaluate_model(self, data):
    if self.outputs is None:
      raise ValueError('Model not built yet')

    if self._session is None:
      self.launch_model(overwrite=False)

    if not self.metric_is_accuracy:
      raise ValueError('Currently this only supports accuracy')

    accuracy = self._session.run(
      self._metric, feed_dict=self._get_default_feed_dict(
        data, is_training=False))
    accuracy *= 100

    console.show_status('Accuracy on test set is {:.2f}%'.format(accuracy))




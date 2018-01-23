from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import config
from tframe import pedia
from tframe import with_graph

from tframe.layers import Activation
from tframe.models.sl.predictor import Predictor
from tframe.utils import console
from tframe.utils.tfdata import TFData


class Classifier(Predictor):
  def __init__(self, mark=None):
    Predictor.__init__(self, mark)
    self._sum_train_acc = None
    self._sum_val_acc = None
    self._probabilities = None


  @with_graph
  def build(self, loss='cross_entropy', optimizer=None):
    # TODO: ... do some compromise
    config.block_validation = True
    Predictor.build(self, loss, optimizer, metric='accuracy',
                    metric_name=pedia.Accuracy)

    self._sum_train_acc = tf.summary.scalar('train_acc', self._metric)
    self._sum_val_acc = tf.summary.scalar('val_acc', self._metric)

    # Find last layer
    if (isinstance(self.last_function, Activation)
        and self.last_function.abbreviation == 'softmax'):
      self._probabilities = self.outputs
    else:
      self._probabilities = tf.nn.softmax(self.outputs, name='possibilities')


  def _update_model(self, data_batch, **kwargs):
    feed_dict = self._get_default_feed_dict(data_batch, is_training=True)

    self._session.run(self._train_step, feed_dict=feed_dict)

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
    feed_dict = self._get_default_feed_dict(
      data_batch, is_training=False)
    sum_train_acc, train_acc = self._session.run(
      [self._sum_train_acc, self._metric], feed_dict=feed_dict)

    feed_dict = self._get_default_feed_dict(
      self._validation_set, is_training=False)
    sum_val_acc, val_acc = self._session.run(
      [self._sum_val_acc, self._metric], feed_dict=feed_dict)

    self._summary_writer.add_summary(sum_train_acc, self._counter)
    self._summary_writer.add_summary(sum_val_acc, self._counter)

    info_dict['Training Accuracy'] = train_acc
    info_dict['Validation Accuracy'] = val_acc

    # Call predictor's _print_progress
    Predictor._print_progress(self, epc, start_time, info_dict)


  def evaluate_model(self, data, with_false=False):
    if self.outputs is None:
      raise ValueError('Model not built yet')
    if self._session is None:
      self.launch_model(overwrite=False)
    if not self.metric_is_accuracy:
      raise ValueError('Currently this only supports accuracy')

    possibilities, accuracy = self._session.run(
      [self._probabilities, self._metric],
      feed_dict=self._get_default_feed_dict(data, is_training=False))
    accuracy *= 100

    console.show_status('Accuracy on test set is {:.2f}%'.format(accuracy))

    if with_false:
      assert isinstance(data, TFData)
      predictions = np.argmax(possibilities, axis=1).squeeze()
      data.update(predictions=predictions)
      labels = data.scalar_labels
      false_indices = [i for i in range(data.sample_num)
                      if predictions[i] != labels[i]]

      from tframe import ImageViewer
      vr = ImageViewer(data[false_indices])
      vr.show()










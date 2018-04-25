from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import hub
from tframe import pedia

from tframe.core.decorators import with_graph
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
  def _build(self, loss='cross_entropy', optimizer=None, *args):
    # TODO: ... do some compromise
    hub.block_validation = True
    # Call parent's method to build using the default loss function
    #  -- cross entropy
    Predictor._build(self, loss, optimizer, metric='accuracy',
                     metric_name=pedia.Accuracy)

    self._sum_train_acc = tf.summary.scalar('train_acc', self._metric)
    self._sum_val_acc = tf.summary.scalar('val_acc', self._metric)

    # Find last layer
    if (isinstance(self.last_function, Activation)
        and self.last_function.abbreviation == 'softmax'):
      self._probabilities = self._outputs
    else:
      self._probabilities = tf.nn.softmax(self._outputs, name='probabilities')


  def update_model(self, data_batch, **kwargs):
    feed_dict = self._get_default_feed_dict(data_batch, is_training=True)

    self._session.run(self._train_step, feed_dict=feed_dict)

    return {}


  def evaluate_model(self, data, with_false=False):
    if self._outputs is None:
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


  def classify(self, data):
    if self._outputs is None:
      raise ValueError('Model not built yet')
    if self._session is None:
      self.launch_model(overwrite=False)

    possibilities = self._session.run(
      self._probabilities, feed_dict=self._get_default_feed_dict(
        data, is_training=False))

    return np.argmax(possibilities, axis=1).squeeze()










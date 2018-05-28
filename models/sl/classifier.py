from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import hub
from tframe import pedia

from tframe.core.decorators import with_graph
from tframe.core import TensorSlot, Group
from tframe.layers import Activation
from tframe.models.sl.predictor import Predictor
from tframe.utils import console
# from tframe.utils.tfdata import TFData
from tframe import DataSet
import tframe.utils.misc as misc

from tframe.models.feedforward import Feedforward


class Classifier(Predictor):
  def __init__(self, mark=None, net_type=Feedforward):
    Predictor.__init__(self, mark, net_type)
    # Private attributes
    self._probabilities = TensorSlot(self, 'Probability')
    self._evaluation_group = Group(self, self._metric, self._probabilities,
                                   name='evaluation group')

  @with_graph
  def _build(self, loss='cross_entropy', optimizer=None, *args):
    # TODO: ... do some compromise
    hub.block_validation = True
    # Call parent's method to build using the default loss function
    #  -- cross entropy
    Predictor._build(self, loss, optimizer, metric='accuracy',
                     metric_name=pedia.Accuracy, metric_is_like_loss=False)
    assert self.outputs.activated

    # Plug tensor into probabilities slot
    output_tensor = self.outputs.tensor
    if not (isinstance(self.last_function, Activation)
        and self.last_function.abbreviation == 'softmax'):
      output_tensor = tf.nn.softmax(output_tensor, name='probabilities')
    self._probabilities.plug(output_tensor)


  def evaluate_model(self, data, export_false=False):
    # Sanity check
    self._sanity_check_before_use(data)

    if not self.metric.symbol == pedia.Accuracy:
      raise ValueError('!! metric must be accuracy')

    # Run group
    feed_dict = self._get_default_feed_dict(data, False)
    result_dict = self._evaluation_group.run(feed_dict=feed_dict)
    probabilities = result_dict[self._probabilities]
    accuracy = result_dict[self._metric] * 100

    console.show_status(
      'Accuracy on {} is {:.2f}%'.format(data.name, accuracy),
      symbol='[Evaluation]')

    if export_false:
      assert isinstance(data, DataSet) and data.targets is not None
      predictions = misc.convert_to_dense_labels(probabilities)
      data.data_dict[pedia.predictions] = predictions
      labels = misc.convert_to_dense_labels(data.targets)
      false_indices = [i for i in range(data.size)
                      if predictions[i] != labels[i]]

      from tframe.data.images.image_viewer import ImageViewer
      vr = ImageViewer(data[false_indices])
      vr.show()


  def classify(self, data):
    # Sanity check
    self._sanity_check_before_use(data)

    feed_dict = self._get_default_feed_dict(data, is_training=False)
    probabilities = self._probabilities.run(feed_dict=feed_dict)

    return misc.convert_to_dense_labels(probabilities)










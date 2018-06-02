from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import hub
from tframe import pedia
from tframe import InputTypes

from tframe.core.decorators import with_graph
from tframe.core import TensorSlot, Group
from tframe.layers import Activation
from tframe.models.sl.predictor import Predictor
from tframe.utils import console
from tframe import DataSet
from tframe.data.base_classes import TFRData
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
  def build(self, optimizer=None, metric='accuracy', **kwargs):
    Predictor.build(self, optimizer=optimizer, loss='cross_entropy',
                    metric=metric, metric_is_like_loss=False,
                    metric_name='Accuracy')


  def _build(self, optimizer=None, metric=None, **kwargs):
    # TODO: ... do some compromise
    hub.block_validation = True
    # If last layer is not softmax layer, add it to model
    if not (isinstance(self.last_function, Activation)
            and self.last_function.abbreviation == 'softmax'):
      self.add(Activation('softmax'))
    # Call parent's method to build using the default loss function
    #  -- cross entropy
    Predictor._build(self, optimize=optimizer, metric=metric, **kwargs)
    assert self.outputs.activated
    # Plug tensor into probabilities slot
    self._probabilities.plug(self.outputs.tensor)


  def evaluate_model(self, data, batch_size=None, extractor=None,
                     export_false=False, **kwargs):
    # Feed data set into model and get results
    false_sample_list = []
    false_label_list = []
    true_label_list = []
    num_samples = 0
    for batch in self.get_data_batches(data, batch_size):
      assert isinstance(batch, DataSet) and batch.targets is not None
      num_samples += len(data.targets)
      # Get predictions
      preds = self._classify_batch(batch, extractor)
      # Select false samples
      true_labels = misc.convert_to_dense_labels(data.targets)
      if len(true_labels) < len(preds):
        true_labels = np.concatenate((true_labels,) * len(preds))
      false_indices = np.argwhere(preds != true_labels).squeeze()
      false_sample_list.append(batch.features[false_indices])
      false_label_list.append(preds[false_indices])
      true_label_list.append(true_labels[false_indices])

    # Show accuracy
    accuracy = (num_samples - len(false_sample_list)) / num_samples * 100
    console.supplement('Accuracy on {} is {:.2f}%'.format(data.name, accuracy))

    # Try to export false samples
    if export_false:
      false_set = DataSet(features=np.concatenate(false_sample_list),
                          targets=np.concatenate(true_label_list))
      false_set.data_dict[pedia.predictions] = np.concatenate(false_label_list)
      from tframe.data.images.image_viewer import ImageViewer
      vr = ImageViewer(false_set)
      vr.show()


  def classify(self, data, batch_size=None, extractor=None):
    predictions = []
    for batch in self.get_data_batches(data, batch_size):
      preds = self._classify_batch(batch, extractor)
      predictions.append(preds)
    return np.concatenate(predictions)


  def _classify_batch(self, batch, extractor):
    assert isinstance(batch, DataSet) and batch.features is not None
    feed_dict = self._get_default_feed_dict(batch, is_training=False)
    probs = self._probabilities.run(feed_dict)
    preds = misc.convert_to_dense_labels(probs)
    if extractor is not None: preds = extractor(preds)
    return preds

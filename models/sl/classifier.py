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
    console.show_status('Evaluating classifier ...')
    for batch in self.get_data_batches(data, batch_size):
      assert isinstance(batch, DataSet) and batch.targets is not None
      # Get predictions
      preds = self._classify_batch(batch, extractor)
      # Get true labels in dense format
      if batch.targets.shape[-1] > 1:
        targets = batch.targets.reshape(-1, batch.targets.shape[-1])
      else: targets = batch.targets
      num_samples += len(targets)
      true_labels = misc.convert_to_dense_labels(targets)
      if len(true_labels) < len(preds):
        assert len(true_labels) == 1
        true_labels = np.concatenate((true_labels,) * len(preds))
      # Select false samples
      false_indices = np.argwhere(preds != true_labels)
      if false_indices.size == 0: continue
      features = batch.features
      if self.input_type is InputTypes.RNN_BATCH:
        features = np.reshape(features, [-1, *features.shape[2:]])
      false_indices = np.reshape(false_indices, false_indices.size)
      false_sample_list.append(features[false_indices])
      false_label_list.append(preds[false_indices])
      true_label_list.append(true_labels[false_indices])
    # Concatenate
    if len(false_sample_list) > 0:
      false_sample_list = np.concatenate(false_sample_list)
      false_label_list = np.concatenate(false_label_list)
      true_label_list = np.concatenate(true_label_list)

    # Show accuracy
    accuracy = (num_samples - len(false_sample_list)) / num_samples * 100
    console.supplement('Accuracy on {} is {:.2f}%'.format(data.name, accuracy))

    # Try to export false samples
    if export_false and accuracy < 100:
      false_set = DataSet(features=false_sample_list, targets=true_label_list)
      false_set.data_dict[pedia.predictions] = false_label_list
      from tframe.data.images.image_viewer import ImageViewer
      vr = ImageViewer(false_set)
      vr.show()


  def classify(self, data, batch_size=None, extractor=None):
    predictions = []
    for batch in self.get_data_batches(data, batch_size):
      preds = self._classify_batch(batch, extractor)
      if isinstance(preds, int): preds = [preds]
      predictions.append(preds)
      if batch.targets is not None:
        # truth = misc.convert_to_dense_labels(np.reshape(
        #   batch.targets, (-1, batch.targets.shape[2])))
        whatever = 1
    return np.concatenate(predictions)


  def _classify_batch(self, batch, extractor):
    assert isinstance(batch, DataSet) and batch.features is not None
    batch = self._sanity_check_before_use(batch)
    feed_dict = self._get_default_feed_dict(batch, is_training=False)
    probs = self._probabilities.run(feed_dict)
    if self.input_type is InputTypes.RNN_BATCH:
      assert len(probs.shape) == 3
      probs = np.reshape(probs, (-1, probs.shape[2]))
    if extractor is None: preds = misc.convert_to_dense_labels(probs)
    else: preds = extractor(probs)
    return preds

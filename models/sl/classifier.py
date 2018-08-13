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
  def build(
      self, optimizer=None, metric='accuracy',  loss='cross_entropy', **kwargs):
    Predictor.build(self, optimizer=optimizer, loss=loss,
                    metric=metric, metric_is_like_loss=False,
                    metric_name='Accuracy', **kwargs)

  def _build(self, optimizer=None, metric=None, **kwargs):
    # TODO: ... do some compromise
    hub.block_validation = True

    # If last layer is not softmax layer, add it to model TODO
    # if not (isinstance(self.last_function, Activation)
    #         and self.last_function.abbreviation == 'softmax'):
    #   self.add(Activation('softmax'))

    # Call parent's method to build using the default loss function
    #  -- cross entropy
    Predictor._build(self, optimize=optimizer, metric=metric, **kwargs)
    assert self.outputs.activated
    # Plug tensor into probabilities slot
    self._probabilities.plug(self.outputs.tensor)

  @with_graph
  def evaluate_model(self, data, batch_size=None, extractor=None,
                     export_false=False, **kwargs):
    console.show_status('Evaluating classifier ...')
    results = self._batch_evaluation(
      self.metric_foreach, data, batch_size, extractor)
    if self.input_type is InputTypes.RNN_BATCH:
      results = np.concatenate([y.flatten() for y in results])
    accuracy = np.mean(results) * 100

    # Show accuracy
    console.supplement('Accuracy on {} is {:.2f}%'.format(data.name, accuracy))

    # export_false option is valid for images only
    if export_false and accuracy < 100.0:
      assert self.input_type is InputTypes.BATCH
      assert isinstance(data, DataSet)
      assert data.features is not None and data.targets is not None

      preds = self.classify(data, batch_size, extractor)

      false_indices = np.argwhere(results == 0).flatten()
      false_features = data.features[false_indices]
      false_targets = data.targets[false_indices]
      false_preds = preds[false_indices]

      false_set = DataSet(false_features, false_targets, **data.properties)
      false_set.properties[pedia.predictions] = false_preds

      from tframe.data.images.image_viewer import ImageViewer
      vr = ImageViewer(false_set)
      vr.show()

  @with_graph
  def classify(self, data, batch_size=None, extractor=None, return_probs=False):
    probs = self._batch_evaluation(
      self._probabilities.tensor, data, batch_size, extractor)
    if return_probs: return probs

    if self.input_type is InputTypes.RNN_BATCH:
      preds = [np.argmax(p, axis=-1) for p in probs]
    else: preds = np.argmax(probs, axis=-1)

    return preds


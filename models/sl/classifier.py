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
from tframe.trainers.metric_slot import MetricSlot


class Classifier(Predictor):
  model_name = 'classifier'

  def __init__(self, mark=None, net_type=Feedforward):
    Predictor.__init__(self, mark, net_type)
    # Private attributes
    self._probabilities = TensorSlot(self, 'Probability')
    # TODO: to be deprecated
    # self._evaluation_group = Group(self, self._metric, self._probabilities,
    #                                name='evaluation group')

  @with_graph
  def build(self, optimizer=None, loss='cross_entropy', metric='accuracy',
            batch_metric=None, eval_metric=None, **kwargs):
    Predictor.build(
      self, optimizer=optimizer, loss=loss, metric=metric,
      batch_metric=batch_metric, eval_metric=eval_metric, **kwargs)

  def _build(self, optimizer=None, metric=None, **kwargs):
    # TODO: ... do some compromise
    hub.block_validation = True

    # If last layer is not softmax layer, add it to model TODO
    # if not (isinstance(self.last_function, Activation)
    #         and self.last_function.abbreviation == 'softmax'):
    #   self.add(Activation('softmax'))

    # Call parent's method to build using the default loss function
    #  -- cross entropy
    Predictor._build(self, optimizer=optimizer, metric=metric, **kwargs)
    assert self.outputs.activated
    # Plug tensor into probabilities slot
    self._probabilities.plug(self.outputs.tensor)

  @with_graph
  def evaluate_model(self, data, batch_size=None, extractor=None,
                     export_false=False, **kwargs):
    # If not necessary, use Predictor's evaluate_model method
    metric_is_accuracy = self.eval_metric.name.lower() == 'accuracy'
    if not export_false or not metric_is_accuracy:
      result = super().evaluate_model(data, batch_size, **kwargs)
      if metric_is_accuracy: result *= 100
      return result

    console.show_status('Evaluating classifier on {} ...'.format(data.name))

    acc_slot = self.metrics_manager.get_slot_by_name('accuracy')
    assert isinstance(acc_slot, MetricSlot)
    acc_foreach = acc_slot.quantity_definition.quantities
    results = self.evaluate(acc_foreach, data, batch_size, extractor,
                            verbose=hub.val_progress_bar)
    if self.input_type is InputTypes.RNN_BATCH:
      results = np.concatenate([y.flatten() for y in results])
    accuracy = np.mean(results) * 100

    # Show accuracy
    console.supplement('Accuracy on {} is {:.3f}%'.format(data.name, accuracy))

    # export_false option is valid for images only
    if export_false and accuracy < 100.0:
      assert self.input_type is InputTypes.BATCH
      assert isinstance(data, DataSet)
      assert data.features is not None and data.targets is not None
      top_k = hub.export_top_k if hub.export_top_k > 0 else 3

      probs = self.classify(data, batch_size, extractor, return_probs=True)
      probs_sorted = np.fliplr(np.sort(probs, axis=-1))
      class_sorted = np.fliplr(np.argsort(probs, axis=-1))
      preds = class_sorted[:, 0]

      false_indices = np.argwhere(results == 0).flatten()
      false_preds = preds[false_indices]

      probs_sorted = probs_sorted[false_indices, :top_k]
      class_sorted = class_sorted[false_indices, :top_k]
      false_set = data[false_indices]

      false_set.properties[pedia.predictions] = false_preds
      false_set.properties[pedia.top_k_label] = class_sorted
      false_set.properties[pedia.top_k_prob] = probs_sorted

      from tframe.data.images.image_viewer import ImageViewer
      vr = ImageViewer(false_set)
      vr.show()

    # Return accuracy
    return accuracy

  @with_graph
  def classify(self, data, batch_size=None, extractor=None,
               return_probs=False, verbose=False):
    probs = self.evaluate(
      self._probabilities.tensor, data, batch_size, extractor, verbose=verbose)
    if return_probs: return probs

    if self.input_type is InputTypes.RNN_BATCH:
      preds = [np.argmax(p, axis=-1) for p in probs]
    else: preds = np.argmax(probs, axis=-1)

    return preds


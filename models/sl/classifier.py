from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import hub
from tframe import pedia
from tframe import InputTypes

from tframe.core.decorators import with_graph
from tframe.core import TensorSlot
from tframe.models.sl.predictor import Predictor
from tframe.utils import console
from tframe.utils.maths.confusion_matrix import ConfusionMatrix
from tframe import DataSet

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
  def classify(self, data, batch_size=None, extractor=None,
               return_probs=False, verbose=False):
    probs = self.evaluate(
      self._probabilities.tensor, data, batch_size, extractor, verbose=verbose)
    if return_probs: return probs

    # TODO: make clear data shape here and add comments
    if self.input_type is InputTypes.RNN_BATCH:
      preds = [np.argmax(p, axis=-1) for p in probs]
    else: preds = np.argmax(probs, axis=-1)

    return preds


  @with_graph
  def evaluate_model(self, data, batch_size=None, extractor=None, **kwargs):
    """This method is a mess."""
    if hub.take_down_confusion_matrix:
      # TODO: (william) please refactor this method
      cm = self.evaluate_pro(
        data, batch_size, verbose=kwargs.get('verbose', False),
        show_class_detail=True, show_confusion_matrix=True)
      # Take down confusion matrix
      from tframe import context
      agent = context.trainer.model.agent
      agent.take_notes('Confusion Matrix on {}:'.format(data.name), False)
      agent.take_notes('\n' + cm.matrix_table().content)
      agent.take_notes('Evaluation Result on {}:'.format(data.name), False)
      agent.take_notes('\n' + cm.make_table().content)
      return cm.accuracy

    # If not necessary, use Predictor's evaluate_model method
    metric_is_accuracy = self.eval_metric.name.lower() == 'accuracy'
    if not metric_is_accuracy:
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

    # Return accuracy
    return accuracy


  @with_graph
  def evaluate_pro(self, data_set, batch_size=None, verbose=False, **kwargs):
    """Evaluate model and give results calculated based on TP, TN, FP, and FN.
    'extractor' is not considered cuz the developer forgot what is this used
    for.
    :param data_set: an instance of dataset contains at least features
    :param batch_size: set this value if your (G)RAM is not big enough to
                       handle the whole dataset
    :param verbose: whether to show status or progress bar stuff
    :param kwargs: other options which will be recognized by PyCharm
    """
    # Get options
    show_confusion_matrix = kwargs.get('show_confusion_matrix', False)
    plot_confusion_matrix = kwargs.get('plot_confusion_matrix', False)
    show_class_detail = kwargs.get('show_class_detail', False)
    export_false = kwargs.get('export_false', False)
    top_k = kwargs.get('export_top_k', 3)

    # Check data_set before get model prediction
    assert isinstance(data_set, DataSet)

    # The assertions below is removed for RNN models
    # assert self.input_type is InputTypes.BATCH
    # assert data_set.features is not None and data_set.targets is not None

    # -------------------------------------------------------------------------
    #  Calculate predicted classes and corresponding probabilities
    # -------------------------------------------------------------------------
    probs = self.classify(
      data_set, batch_size, return_probs=True, verbose=verbose)
    # This provides necessary information for image viewer presentation
    # i.e., the sorted probabilities for each class
    probs_sorted = np.fliplr(np.sort(probs, axis=-1))
    class_sorted = np.fliplr(np.argsort(probs, axis=-1))
    preds = class_sorted[:, 0]
    truths = np.ravel(data_set.dense_labels)

    # Apply batch mask if provided
    if pedia.batch_mask in data_set.data_dict:
      mask = np.array(data_set.data_dict[pedia.batch_mask]).astype(bool)
      mask = np.ravel(mask)
      preds = preds[mask]
      truths = truths[mask]

    # Produce confusion matrix
    cm = ConfusionMatrix(
      num_classes=data_set.num_classes,
      class_names=data_set.properties.get(pedia.classes, None))
    cm.fill(preds, truths)

    # Print evaluation results
    if show_confusion_matrix:
      console.show_info('Confusion Matrix:')
      console.write_line(cm.matrix_table(kwargs.get('cell_width', None)))
    if plot_confusion_matrix: cm.sklearn_plot()
    console.show_info(f'Evaluation Result ({data_set.name}):')
    console.write_line(cm.make_table(
      decimal=4, class_details=show_class_detail))

    # Visualize false set if specified
    if export_false:
      indices = np.argwhere(preds != truths).flatten()
      false_set = data_set[indices]
      false_set.properties[pedia.predictions] = preds[indices]
      false_set.properties[pedia.top_k_label] = class_sorted[indices, :top_k]
      false_set.properties[pedia.top_k_prob] = probs_sorted[indices, :top_k]
      return cm, false_set
    else: return cm


  def evaluate_image_sets(
      self, *data_sets, visualize_last_false_set=False, **kwargs):
    """Evaluate image datasets. This method is a generalization of what usually
       appears in the `core` module of an image classification task. """
    from tframe.data.images.image_viewer import ImageViewer

    configs = {'show_class_detail': True, 'show_confusion_matrix': True}
    configs.update(kwargs)

    for i, data_set in enumerate(data_sets):
      if i == len(data_sets) - 1 and visualize_last_false_set:
        _, false_set = self.evaluate_pro(
          data_set, batch_size=hub.eval_batch_size,
          export_false=visualize_last_false_set, export_top_k=hub.export_top_k,
          **configs)
        # Visualize false set
        viewer = ImageViewer(false_set)
        viewer.show()
      else:
        self.evaluate_pro(data_set, batch_size=hub.eval_batch_size, **configs)

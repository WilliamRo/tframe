from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from tframe import tf

from tframe.core.decorators import with_graph

from tframe.models.model import Model
from tframe.nets.net import Net
from tframe import pedia, checker, context
from tframe.data.dataset import DataSet


class Feedforward(Model, Net):
  """Feedforward network, also known as multilayer perceptron"""
  model_name = 'MLP'

  def __init__(self, mark=None):
    Model.__init__(self, mark)
    Net.__init__(self, 'FeedforwardNet')
    self.superior = self
    self._default_net = self

  @with_graph
  def _build(self, **kwargs):
    from tframe import hub as th
    from tframe import mu as m
    from tframe import console

    # Feed forward to get outputs
    output = self()

    if self._inter_type == pedia.fork: return

    assert isinstance(output, tf.Tensor)
    self.outputs.plug(output)

    # val_outputs is designed for models whose training data have different
    # size with actual data fed in non-training situations such as validation
    if th.non_train_input_shape is None:
      self.val_outputs.plug(output)
      return

    # Create val_output tensor
    val_input = m.Input(sample_shape=th.non_train_input_shape,
                        name=pedia.non_train_input)
    val_output = self(val_input())
    self.val_outputs.plug(val_output)
    self._shadow_input = val_input

    console.show_status(
      f'Shadow output with input shape {th.non_train_input_shape} has been created.')

  # region : Public Methods

  @staticmethod
  def get_tensor_to_export(trainer):
    """Used in trainer._take_notes_for_export"""
    from tframe.trainers.trainer import Trainer
    assert isinstance(trainer, Trainer)

    tensors = OrderedDict()
    num = checker.check_positive_integer(trainer.th.sample_num)
    # .. fetch tensors
    fetches_dict = context.tensors_to_export
    if len(fetches_dict) == 0: return tensors
    results = trainer.model.evaluate(
      list(fetches_dict.values()), trainer.validation_set[:num])

    exemplar_names = []
    for i in range(num):
      name = 'Exemplar {}'.format(i)
      tensors[name] = OrderedDict()
      exemplar_names.append(name)

    # .. fill tensor_dict
    for i, array_list in enumerate(results):
      tensor_name = list(fetches_dict.keys())[i]
      for j, array in enumerate(array_list):
        if j < num: tensors[exemplar_names[j]][tensor_name] = array

    return tensors

  # endregion : Public Methods

  # region : Abstract Implementations

  def _evaluate_batch(self, fetch_list, data_batch, **kwargs):
    # Sanity check
    assert isinstance(fetch_list, list)
    checker.check_fetchable(fetch_list)
    assert isinstance(data_batch, DataSet)

    # Run session
    feed_dict = self._get_default_feed_dict(data_batch, is_training=False)
    batch_outputs = self.session.run(fetch_list, feed_dict)

    return batch_outputs

  # endregion : Abstract Implementations

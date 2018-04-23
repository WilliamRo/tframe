from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tframe as tfr


class Slot(object):
  """Slots exist in tframe models. Once plugged in with tensorflow op
    during model building stage, they are called 'activated'."""
  op_classes = []

  def __init__(self, model, name):
    assert isinstance(model, tfr.models.Model)
    self._model = model
    self._op = None
    self.name = name

  # region : Properties

  @property
  def activated(self):
    return self._op is not None

  @property
  def op(self):
    return self._op

  # endregion : Properties

  # region : Public Methods

  def plug(self, op, **kwargs):
    if op.__class__ not in self.op_classes:
      raise TypeError('!! op should be in {}'.format(self.op_classes))
    self._op = op

  def substitute(self, op):
    self.plug(op)

  def run(self, fetches=None, feed_dict=None):
    if not self.activated:
      raise AssertionError('!! This slot is not activated')

    fetches = self._op if fetches is None else fetches
    if not isinstance(fetches, (tuple, list)): fetches = [fetches]
    assert isinstance(fetches, (list, tuple))
    ops = []
    for entity in fetches:
      op = entity
      if isinstance(op, Slot): op = entity.op
      elif op.__class__ not in [
        tf.Tensor, tf.Operation, tf.summary.Summary, tf.Variable]:
        raise TypeError('!! Unknown type {}'.format(op.__class__))
      ops.append(op)

    with self._model.graph.as_default():
      return self._model.session.run(ops, feed_dict=feed_dict)

  def fetch(self, feed_dict=None):
    result = self.run(feed_dict=feed_dict)
    if isinstance(result, (list, tuple)):
      result = result[0]
    return result

  # endregion : Public Methods


class TensorSlot(Slot):
  op_classes = [tf.Tensor]
  def __init__(self, model, name='tensor'):
    super().__init__(model, name)

  # region : Properties

  @property
  def tensor(self):
    return self._op

  @property
  def shape_list(self):
    if not self.activated:
      raise ValueError('!! slot has not been activated yet')
    assert isinstance(self._op, tf.Tensor)
    return self._op.shape.as_list()

  @property
  def dtype(self):
    if not self.activated:
      raise ValueError('!! slot has not been activated yet')
    assert isinstance(self._op, tf.Tensor)
    return self._op.dtype

  # endregion : Properties

  def plug(self, op, collection=None):
    super().plug(op)
    # Add to tensorflow collection
    if collection is not None:
      tf.add_to_collection(collection, self._op)


class SummarySlot(Slot):
  op_classes = [tf.Tensor]
  def __init__(self, model, name='summary'):
    super().__init__(model, name)

  @property
  def summary(self):
    return self._op


class OperationSlot(Slot):
  op_classes = [tf.Operation]
  def __init__(self, model, name='operation'):
    super().__init__(model, name)

  @property
  def operation(self):
    return self._op


class VariableSlot(TensorSlot):
  op_classes = [tf.Variable]
  # TODO: need to be more appropriate
  _null_value = -1.0
  def __init__(self, model, name='variable'):
    super().__init__(model, name)

  @property
  def variable(self):
    return self._op

  @property
  def never_assigned(self):
    return self.fetch() == self._null_value

  def assign(self, value):
    self.run(tf.assign(self._op, value))

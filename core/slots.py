from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tframe as tfr
from .quantity import Quantity


class Slot(object):
  """Slots exist in tframe models. Once plugged in with tensorflow op
    during model building stage, they are called 'activated'."""
  op_classes = []

  def __init__(self, model, name):
    assert isinstance(model, tfr.models.Model)
    self._model = model
    self._op = None
    self.name = name
    self.sleep = False

  # region : Properties

  @property
  def activated(self):
    return self._op is not None

  @property
  def op(self):
    return self._op

  # endregion : Properties

  # region : Overriding

  def __str__(self):
    return self.name

  # endregion : Overriding

  # region : Public Methods

  def plug(self, op, **kwargs):
    if not isinstance(op, tuple(self.op_classes)):
      raise TypeError('!! op should be in {}'.format(self.op_classes))
    self._op = op

  def substitute(self, op):
    if op.__class__ not in self.op_classes:
      raise TypeError('!! op should be in {}'.format(self.op_classes))
    self._op = op

  def run(self, feed_dict=None):
    return self._model.session.run(self._op, feed_dict=feed_dict)

  # TODO: when everything is settled, remove this method
  def run_(self, fetches=None, feed_dict=None):
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
        # Nested tensor slots shall not be run
        raise TypeError('!! Unknown type {}'.format(op.__class__))
      ops.append(op)

    with self._model.graph.as_default():
      result = self._model.session.run(ops, feed_dict=feed_dict)
      if isinstance(result, (tuple, list)): result = result[0]
      return result

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
    self._quantity_definition = None

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

  @property
  def quantity_definition(self):
    if self._quantity_definition is None:
      raise ValueError('!! {} does not have quantity definition.'.format(
        self.name))
    assert isinstance(self._quantity_definition, Quantity)
    return self._quantity_definition

  # endregion : Properties

  def plug(self, op, collection=None, quantity_def=None):
    super().plug(op)
    # Add to tensorflow collection
    if collection is not None:
      tf.add_to_collection(collection, self._op)
    if quantity_def is not None: assert isinstance(quantity_def, Quantity)
    self._quantity_definition = quantity_def


class NestedTensorSlot(Slot):
  op_classes = [tf.Tensor]
  
  @property
  def nested_tensors(self):
    return self._op

  def plug(self, op, **kwargs):
    self._check_op(op)
    self._op = op

  @staticmethod
  def _check_op(entity):
    if isinstance(entity, (list, tuple)):
      for obj in entity: NestedTensorSlot._check_op(obj)
    elif entity.__class__ not in NestedTensorSlot.op_classes:
      raise TypeError('!! Unknown object found during plugging')


class SummarySlot(Slot):
  op_classes = [tf.Tensor]
  def __init__(self, model, name='summary'):
    super().__init__(model, name)

  @property
  def summary(self):
    return self._op


class IndependentSummarySlot(SummarySlot):
  def __init__(self, model, name='batch_summary'):
    super().__init__(model, name)
    # Attributes
    self._mascot = None
    self._op = None

  def write(self, val):
    # Check tensor
    if self._mascot is None:
      self._mascot = tf.placeholder(dtype=tf.float32)
      self._op = tf.summary.scalar(
        self.name, self._mascot, collections=[tfr.pedia.invisible])
    assert self.activated
    summ = self._model.session.run(self.summary, feed_dict={self._mascot: val})
    self._model.agent.write_summary(summ)


class OperationSlot(Slot):
  op_classes = [tf.Operation, tf.Tensor]
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
    self._model.session.run(tf.assign(self._op, value))

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

import tframe as tfr
from .quantity import Quantity
from .function import Function


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

    # This attribute is designed for complicated metrics such as IoU scores
    # for instance segmentation task (2021-12-10 @ CUHK RRSSB 212A-06)
    # A post_processor should have the following signature:
    #   def post_processor(array: np.ndarray, data: tframe.DataSet)
    self.post_processor = None

  # region : Properties

  @property
  def activated(self):
    return self._op is not None

  @property
  def op(self):
    return self._op

  # endregion : Properties

  # region : Overriding

  def __str__(self): return self.name

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

  def run(self, feed_dict=None, data=None):
    result = self._model.session.run(self._op, feed_dict=feed_dict)
    if callable(self.post_processor):
      result = self.post_processor(result, data)
    return result

  def fetch(self, feed_dict=None, data=None):
    result = self.run(feed_dict=feed_dict, data=data)
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


class OutputSlot(TensorSlot):

  def __init__(self, model, f, loss=None, loss_coef=1.0, name='output',
               target_key=None, last_only=False):
    from tframe import losses  # TODO: refactor this line
    # Sanity check
    assert isinstance(f, Function)
    assert isinstance(loss_coef, float) and loss_coef > 0
    # Currently this class should not appear in RNN codes
    assert not last_only
    # Call parent's constructor
    super().__init__(model, name=name)
    # Other attributes
    self.function = f
    self.loss_coef = loss_coef
    self.loss_quantity = losses.get(loss, last_only=last_only) if loss else None
    self.target_key = target_key
    self.target_slot = TensorSlot(model, '{}_slot'.format(name))
    self.loss_slot = TensorSlot(model, '{}_loss_slot'.format(name))

  @property
  def injection_flag(self):
    return self.loss_quantity is not None and self.loss_coef > 0

  def auto_plug(self):
    super().plug(self.function.output_tensor)
    if not self.loss_quantity: return None
    # Generate loss tensor for error injection
    assert isinstance(self.target_key, str)
    target_tensor = tf.placeholder(
      tfr.hub.dtype, self.shape_list, name=self.target_key)
    # in model.py -> _get_default_feed_dict method, this placeholder will be
    #   matched with corresponding value in data via target_key
    self.target_slot.plug(target_tensor, collection=tfr.pedia.default_feed_dict)
    loss_tensor = tf.multiply(
      self.loss_quantity(self.target_slot.tensor, self.tensor), self.loss_coef)
    self.loss_slot.plug(loss_tensor, quantity_def=self.loss_quantity,
                        collection=tfr.pedia.injection_loss)
    return loss_tensor

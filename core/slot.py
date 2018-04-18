from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tframe as tfr


class Slot(object):
  """Slots exist in tframe models. Once plugged in with tensorflow op
    during model building stage, they are called 'activated'."""
  def __init__(self, model, name):
    assert isinstance(model, tfr.models.Model)
    self._model = model
    self._op = None
    self.name = name

  # region : Properties

  @property
  def activated(self):
    return self._op is not None

  # endregion : Properties

  # region : Public Methods

  def plug(self, op, **kwargs):
    self._op = op

  def run(self, fetches=None, feed_dict=None):
    fetches = self._op if fetches is None else fetches
    if not self.activated:
      raise AssertionError('!! This slot is not activated')
    with self._model.graph.as_default():
      return self._model.session.run(fetches, feed_dict=feed_dict)

  def fetch(self, feed_dict=None):
    return self.run(feed_dict=feed_dict)

  # endregion : Public Methods


class TensorSlot(Slot):
  def __init__(self, model, name='tensor'):
    super().__init__(model, name)

  @property
  def tensor(self):
    return self._op


class SummarySlot(Slot):
  def __init__(self, model, name='summary'):
    super().__init__(model, name)

  @property
  def summary(self):
    return self._op

  def write_summary(self):
    pass


class OperationSlot(Slot):
  def __init__(self, model, name='operation'):
    super().__init__(model, name)

  @property
  def operation(self):
    return self._op


class VariableSlot(TensorSlot):
  def __init__(self, model, name='variable'):
    super().__init__(model, name)
    # Attributes
    self._never_assigned = True

  @property
  def never_assigned(self):
    return self._never_assigned

  def assign(self, value):
    self.run(tf.assign(self._op, value))
    self._never_assigned = False

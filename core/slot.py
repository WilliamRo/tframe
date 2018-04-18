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

  @property
  def op(self):
    return self._op

  # endregion : Properties

  # region : Public Methods

  def plug(self, op, **kwargs):
    self._op = op

  def fetch(self, feed_dict=None):
    if not self.activated:
      raise AssertionError('!! This slot is not activated')
    with self._model.graph.as_default():
      return self._model.session.run(self._op, feed_dict=feed_dict)

  # endregion : Public Methods


class TensorSlot(Slot):
  def __init__(self, model, name='tensor'):
    super().__init__(model, name)


class SummarySlot(Slot):
  def __init__(self, model, name='summary'):
    super().__init__(model, name)

  def write_summary(self):
    pass


class OperationSlot(Slot):
  def __init__(self, model, name='operation'):
    super().__init__(model, name)



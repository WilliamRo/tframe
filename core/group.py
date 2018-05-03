from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tframe as tfr
from tframe import hub

from tframe.core import Slot, TensorSlot, NestedTensorSlot
from tframe.core import SummarySlot, OperationSlot


class Group(object):
  """A Group consists of Slots"""
  def __init__(self, model, *slots, name='group'):
    assert isinstance(model, tfr.models.Model)
    self._model = model
    # Attributes
    self._slots = []
    self._init_slots(slots)
    self.name = name
    # Make sure group has at least one slot
    assert isinstance(self._slots[0], Slot)

  # region : Properties
  # endregion : Properties

  # region : Public Methods

  def run(self, feed_dict=None):
    """Run group in session. Slots except SummarySlot should be activated"""
    fetches = []
    for slot in self._slots:
      if isinstance(slot, SummarySlot) and (
          not slot.activated or not hub.summary): continue
      if not slot.activated:
        raise AssertionError('!! {} must be activated'.format(slot.name))
      if not slot.sleep: fetches.append(slot)

    with self._model.graph.as_default():
      results = self._model.session.run(
        [slot.op for slot in fetches], feed_dict=feed_dict)

    # Check results
    tensor_dict = collections.OrderedDict()
    for slot, val in zip(fetches, results):
      if isinstance(slot, SummarySlot):
        self._model.agent.write_summary(val)
      elif isinstance(slot, (TensorSlot, NestedTensorSlot)):
        tensor_dict[slot] = val

    # Return tensor dictionary
    return tensor_dict

  # endregion : Public Methods

  # region : Private Methods

  def _init_slots(self, slots):
    if len(slots) == 0: raise ValueError('!! not slot found')
    for slot in slots:
      if not isinstance(slot, Slot):
        raise TypeError('!! slot must be an instance of Slot')
      self._slots.append(slot)

  # endregion : Private Methods



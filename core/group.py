from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tframe as tfr

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

  @property
  def tensor_slots(self):
    return [s for s in self._slots if isinstance(s, TensorSlot)]

  # endregion : Properties

  # region : Public Methods

  def run(self, feed_dict=None, allow_sum=True, data=None):
    """Run group in session. Slots except SummarySlot should be activated"""
    fetches = []
    for slot in self._slots:
      # Make sure every slot in this group except summary slot has been
      #  activated
      if not slot.activated and slot.name != 'summary':
        raise AssertionError(f'!! Slot `{slot.name}` not activated')

      if isinstance(slot, SummarySlot) and (
          not slot.activated or not tfr.context.hub.summary or not allow_sum):
        continue
      # if not slot.activated:
      #   raise AssertionError('!! {} must be activated'.format(slot.name))
      if slot.activated and not slot.sleep: fetches.append(slot)

    with self._model.graph.as_default():
      results = self._model.session.run(
        [slot.op for slot in fetches], feed_dict=feed_dict)

    # Check results
    tensor_dict = collections.OrderedDict()
    for slot, val in zip(fetches, results):
      # Do post-process if post_processor is provided
      if callable(slot.post_processor):
        val = slot.post_processor(val, data)

      if isinstance(slot, SummarySlot):
        self._model.agent.write_summary(val)
      elif isinstance(slot, (TensorSlot, NestedTensorSlot)):
        tensor_dict[slot] = val

    # Return tensor dictionary
    return tensor_dict

  def add(self, slot):
    if not isinstance(slot, Slot):
      raise TypeError('!! member added to a group must be a Slot')
    self._slots.append(slot)

  def remove(self, slot):
    assert isinstance(slot, Slot)
    self._slots.remove(slot)

  # endregion : Public Methods

  # region : Private Methods

  def _init_slots(self, slots):
    if len(slots) == 0: raise ValueError('!! not slot found')
    for slot in slots: self.add(slot)

  # endregion : Private Methods



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import hub

from tframe.models.model import Model
from tframe.nets import RNet
from tframe.layers import Input

from tframe.core.decorators import with_graph
from tframe.core import TensorSlot, NestedTensorSlot


class Recurrent(Model, RNet):
  """Recurrent neural network base class"""
  # TODO: where will this variable be used?
  model_name = 'RNN'

  def __init__(self, mark=None):
    Model.__init__(self, mark)
    RNet.__init__(self, 'RecurrentNet')
    self.superior = self
    self._default_net = self
    # Attributes
    self._state_slot = NestedTensorSlot(self, 'State')
    self._grad_buffer_slot = NestedTensorSlot(self, 'GradBuffer')
    # mascot will be initiated as a placeholder with no shape specified
    # .. and will be put into initializer argument of tf.scan
    self._mascot = None

    # TODO: BETA
    self.last_scan_output = None

  # region : Properties

  @property
  def grad_buffer_slot(self):
    return self._grad_buffer_slot

  # endregion : Properties

  # region : Build

  @with_graph
  def _build(self, **kwargs):
    # self.init_state should be called for the first time inside this method
    #  so that it can be initialized within the appropriate graph

    # Do some initialization
    self._mascot = tf.placeholder(dtype=hub.dtype, name='mascot')

    # :: Define output
    # Make sure input has been defined
    if self.input_ is None: raise ValueError('!! input not found')
    assert isinstance(self.input_, Input)
    # Input placeholder has a shape of [batch_size, num_steps, *sample_shape]
    self.input_.set_group_shape((None, None))
    # Transpose input so as to fit the input of tf.scan
    input_placeholder = self.input_()
    assert isinstance(input_placeholder, tf.Tensor)
    perm = list(range(len(input_placeholder.shape.as_list())))
    # elems.shape = [num_steps, batch_size, *sample_shape]
    elems = tf.transpose(input_placeholder, [1, 0] + perm[2:])

    # Pop last softmax if necessary
    last_softmax = self.pop_last_softmax()
    # Call scan to produce a dynamic op
    initializer = self._mascot, self.init_state
    # TODO: BETA
    if hub.use_rtrl:
      initializer += ((self._mascot,) * len(self.repeater_containers),)
      scan_outputs, state_sequences, grads = tf.scan(
        self, elems, initializer=initializer, name='Scan')

      self._grad_tensors = Recurrent._get_last_tensors(grads)
      # dL/dLast_output = None
      # self._last_scan_output = Recurrent._get_last_tensors(scan_outputs)
      self.last_scan_output = scan_outputs
    else:
      scan_outputs, state_sequences = tf.scan(
        self, elems, initializer=initializer, name='Scan')
    # Activate state slot
    assert isinstance(self._state_slot, NestedTensorSlot)

    # Get last state and distribute to all recurrent-child
    last_state = Recurrent._get_last_tensors(state_sequences)
    self._new_state_tensor = last_state
    self._distribute_last_tensors()

    # Plug last state to corresponding slot
    self._state_slot.plug(last_state)
    self._update_group.add(self._state_slot)
    # TODO: BETA
    if hub.use_rtrl: self._update_group.add(self.grad_buffer_slot)

    # Transpose scan outputs to get final outputs
    assert isinstance(scan_outputs, tf.Tensor)
    perm = list(range(len(scan_outputs.shape.as_list())))
    outputs = tf.transpose(scan_outputs, [1, 0] + perm[2:])

    # Apply last softmax if necessary
    if last_softmax is not None:
      self._logits_tensor = outputs
      outputs = last_softmax(outputs)
      # Put last softmax back
      self.add(last_softmax)

    # Output has a shape of [batch_size, num_steps, *output_shape]
    self.outputs.plug(outputs)

  @staticmethod
  def _get_last_tensors(states):
    """This method is used specifically for the tf.scan output"""
    if isinstance(states, (list, tuple)):
      last_state = []
      for obj in states: last_state.append(Recurrent._get_last_tensors(obj))
      return last_state
    else:
      assert isinstance(states, tf.Tensor)
      return states[-1]

  # endregion: Build

















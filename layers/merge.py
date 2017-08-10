from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..layers.layer import Layer
from ..utils import get_scale


class Concatenate(Layer):
  full_name = 'concatenate'
  abbreviation = 'concat'

  def __init__(self, companions=None):
    """
    Initiate a concatenate layer
    :param companions: a dictionary with format:
                       {tensor0: insert_position0, ..., 
                        tensorN: insert_positionN}
                        companion tensors will be inserted into input list
                        at the specific position
    """
    # Check companion
    if not companions is None:
      for key in companions.keys():
        if not isinstance(key, tf.Tensor):
          raise TypeError('key must be a tensor')
        if not isinstance(companions[key], int):
          raise TypeError('value must be an integer')

    self._companions = companions

  def _link(self, inputs, **kwargs):
    if isinstance(inputs, tf.Tensor):
      inputs = [inputs]

    # Avoid that insert operation below changes the original list
    inputs = inputs.copy()

    assert isinstance(inputs, list)
    if not self._companions is None:
      for tensor in self._companions.keys():
        assert isinstance(tensor, tf.Tensor)
        inputs.insert(self._companions[tensor], tensor)

    # Check inputs
    if len(inputs) < 2:
      raise ValueError('inputs to concatenate layer must have a length'
                        ' larger than 1')

    # Prepare inputs for concatenation
    assert isinstance(inputs[0], tf.Tensor)
    leader_shape_tensor = tf.shape(inputs[0])
    leader_shape = inputs[0].get_shape().as_list()

    for i in range(1, len(inputs)):
      assert isinstance(inputs[i], tf.Tensor)
      shape = inputs[i].get_shape().as_list()
      shape_tensor = tf.shape(inputs[i])
      ones = tf.ones(
        [leader_shape_tensor[0]] + leader_shape[1:-1] + [shape[-1]])

      target_shape = ([shape_tensor[0]] + [1]*(len(leader_shape) - 2)
                      + [shape[-1]])
      reshaped = tf.reshape(inputs[i], target_shape)

      inputs[i] = reshaped * ones

    result = tf.concat(inputs, axis=len(leader_shape) - 1)
    self.neuron_scale = get_scale(result)
    return result







from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tframe.nets.net as tfr_net

from tframe import pedia
from tframe.core import single_input

from tframe.layers import Activation
from tframe.layers import Linear
from tframe.layers.layer import Layer


class ResidualNet(tfr_net.Net):
  """Residual learning building block
      Ref: Kaiming He, etc. 'Deep Residual Learning for Image Recognition'.
      https://arxiv.org/abs/1512.03385"""
  def __init__(self, force_transform=False, **kwargs):
    # Call parent's constructor
    tfr_net.Net.__init__(self, 'res', **kwargs)

    # Initialize new fields
    self._force_transform = force_transform
    self._post_processes = []

    self._current_collection = self.children
    self._transform_layer = None


  # region : Properties

  def structure_string(self, detail=True, scale=True):
    body = tfr_net.Net.structure_string(self, detail, scale)
    result = 'sc({}){}'.format(
      body, '' if self._transform_layer is None else 't')

    # Add post process layers
    for layer in self._post_processes:
      assert isinstance(layer, Layer)
      result += '-> {}'.format(self._get_layer_string(layer, scale))

    # Return
    return result

  # endregion : Properties


  # region : Abstract Implementation

  @single_input
  def _link(self, input_, **kwargs):
    """..."""
    assert isinstance(input_, tf.Tensor)

    # Link main part
    output = input_
    for layer in self.children:
      assert isinstance(layer, Layer)
      output = layer(output)

    # Shortcut
    input_shape = input_.get_shape().as_list()
    output_shape = output.get_shape().as_list()
    origin = input_
    if len(input_shape) != len(output_shape):
      raise ValueError('!! input and output must have the same dimension')
    if self._force_transform or input_shape != output_shape:
      if len(input_shape) == 2:
        # Add linear layer
        use_bias = self.kwargs.get('use_bias', False)
        self._transform_layer = Linear(
          output_dim=output_shape[1], use_bias=use_bias)
      else: raise TypeError(
        '!! ResNet in tframe currently only support linear transformation')
      # Save add
      self._transform_layer.full_name = self._get_new_name(
        self._transform_layer)
      origin = self._transform_layer(origin)
    # Do transformation
    output = output + origin

    # Link post process layers
    for layer in self._post_processes:
      assert isinstance(layer, Layer)
      if isinstance(layer, Activation): self._logits_tensor = output
      output = layer(output)

    # Return result
    return output

  # endregion : Abstract Implementation


  # region : Public Methods

  def add_shortcut(self):
    self._current_collection = self._post_processes

  def add(self, layer=None, **kwargs):
    if not isinstance(layer, Layer): raise TypeError(
        '!! layer added to ResNet block must be an instance of Layer')
    assert isinstance(self._current_collection, list)
    name = self._get_new_name(layer)
    layer.full_name = name
    self._current_collection.append(layer)

  # endregion : Public Methods

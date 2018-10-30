from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.nets.net import Net
from tframe.nets.rnet import RNet
from tframe.layers.layer import Layer


class NodeRegister(object):
  """
  """
  def __init__(self, model):
    # Attributes
    assert isinstance(model, Net)
    self._model = model

    self.blocks = []

    self._census()

  # region : Private Methods

  def _census(self):
    for block in self._model.children:
      if isinstance(block, RNet):
        self.blocks.append(Block(block))
      elif isinstance(block, Net):
        for layer in block.children:
          assert isinstance(layer, Layer)
          if layer.is_nucleus:
            self.blocks.append(Block(layer))
      else:
        raise TypeError('!! Unknown block type {}'.format(type(block)))

  # endregion : Private Methods


class Block(object):
  """"""
  def __init__(self, container):
    assert isinstance(container, (RNet, Layer))
    self.container = container
    self.dynamic_nodes = []
    self.variables = []

    self._init_nodes_n_variables()

  @property
  def type(self):
    return type(self.container)

  @property
  def repeater_tensor(self):
    if isinstance(self.container, Layer):
      repeater = self.container.output_tensor
    elif isinstance(self.container, RNet):
      repeater = self.container.post_dynamic_tensor
    else: raise TypeError(
      '!! Unknown container type {}'.format(type(self.container)))
    assert isinstance(repeater, tf.Tensor)
    return repeater

  @property
  def scope(self):
    # The parameters are in the same scope of the repeater
    names = self.repeater_tensor.name.split('/')
    # tensor name always begin with Scan/while/
    return '/'.join(names[2:-1])

  def _init_nodes_n_variables(self):
    """Currently, repeater can not be a tuple or a list
    """
    # Separate all dynamic nodes
    repeater = self.repeater_tensor
    # self.dynamic_nodes = tf.split(
    #   repeater, num_or_size_splits=repeater.shape[1], axis=1)

    # Get all variables
    self.variables = tf.trainable_variables(self.scope)




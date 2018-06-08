from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.core.decorators import with_graph

from tframe.models.model import Model
from tframe.nets.net import Net
from tframe import pedia

from tframe.core import TensorSlot


class Feedforward(Model, Net):
  """Feedforward network, also known as multilayer perceptron"""
  model_name = 'MLP'

  def __init__(self, mark=None):
    Model.__init__(self, mark)
    Net.__init__(self, 'FeedforwardNet')
    self.superior = self
    self._default_net = self

  @with_graph
  def _build(self, **kwargs):
    # Feed forward to get outputs
    output = self()
    if not self._inter_type == pedia.fork:
      assert isinstance(output, tf.Tensor)
      self.outputs.plug(output)


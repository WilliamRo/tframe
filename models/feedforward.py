from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe.core.decorators import with_graph

from tframe.models.model import Model
from tframe.nets.net import Net

from tframe.core import TensorSlot


class Feedforward(Model, Net):
  """Feedforward network, also known as multilayer perceptron"""
  model_name = 'MLP'

  def __init__(self, mark=None):
    Model.__init__(self, mark)
    Net.__init__(self, 'FeedforwardNet')

  @with_graph
  def _build(self):
    # Feed forward to get outputs
    self.outputs.plug(self())


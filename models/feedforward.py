from __future__ import absolute_import

from tframe import with_graph

from tframe.models.model import Model
from tframe.nets.net import Net


class Feedforward(Model, Net):
  """Feedforward network, also known as multilayer perceptron"""
  model_name = 'MLP'

  def __init__(self, mark=None):
    Model.__init__(self, mark)
    Net.__init__(self, 'FeedforwardNet')

  @with_graph
  def build(self):
    # Feed forward to get outputs
    self.outputs = self()

    self._built = True



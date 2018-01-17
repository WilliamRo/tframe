from __future__ import absolute_import

from .model import Model
from ..nets import Net


class Feedforward(Model, Net):
  """Feedforward network, also known as multilayer perceptron"""
  model_name = 'MLP'

  def __init__(self, mark=None):
    Model.__init__(self, mark)
    Net.__init__(self, 'FeedforwardNet')

  def build(self):
    # Feed forward to get outputs
    self.outputs = self()

    self._built = True



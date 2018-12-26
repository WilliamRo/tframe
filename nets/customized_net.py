from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe.nets.net import Net
from tframe.layers.layer import Layer
from tframe.layers.common import Activation


class CustomizedNet(Net):
  net_name = None

  def __init__(self, **kwargs):
    super().__init__(self.net_name, **kwargs)

  def structure_string(self, detail=True, scale=True):
    return self.net_name

  def _link(self, *inputs, **kwargs):
    assert len(inputs) == 1
    output = self.link(inputs[0])
    for f in self.children:
      assert isinstance(f, Layer)
      if isinstance(f, Activation):
        self._logits_tensor = output
      output = f(output)
    return output

  def link(self, inputs):
    raise NotImplemented


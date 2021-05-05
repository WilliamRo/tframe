from lambo.gui.vinci.vinci import DaVinci

from tframe import DataSet
from tframe import Predictor
from tframe.nets.net import Net
from tframe.layers.convolutional import _Conv as ConvLayer


class ConvVisualizer(DaVinci):

  def __init__(self, size: int=5, **kwargs):
    super(ConvVisualizer, self).__init__('Conv Visualizer', size, size)

    self.kwargs = kwargs


  @classmethod
  def feedforward(cls, model: Predictor, data: DataSet):
    conv_layers = cls.extract_conv_layers(model)
    tensors = [l.output_tensor for l in conv_layers]
    outputs = model.evaluate(
      fetches=tensors, data=data, batch_size=1, verbose=True)


  @classmethod
  def extract_conv_layers(cls, net: Net):
    layers = []
    for func in net.children:
      if isinstance(func, Net): layers.extend(cls.extract_conv_layers(func))
      elif isinstance(func, ConvLayer): layers.append(func)
    return layers



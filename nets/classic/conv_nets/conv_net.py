from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe.nets.net import Net

from tframe.layers.advanced import Dense

from tframe.layers.common import Activation
from tframe.layers.common import Dropout
from tframe.layers.common import Flatten
from tframe.layers.common import Input
from tframe.layers.common import Linear

from tframe.layers.convolutional import Conv1D
from tframe.layers.convolutional import Conv2D
from tframe.layers.convolutional import Deconv2D

from tframe.layers.highway import LinearHighway

from tframe.layers.hyper.dense import Dense as HyperDense

from tframe.layers.merge import Merge
from tframe.layers.merge import ShortCut

from tframe.layers.normalization import BatchNormalization
from tframe.layers.normalization import LayerNormalization

from tframe.layers.preprocess import Normalize

from tframe.layers.pooling import AveragePooling2D
from tframe.layers.pooling import GlobalAveragePooling2D
from tframe.layers.pooling import MaxPool2D


class ConvNet(object):

  def add_to(self, model):
    assert isinstance(model, Net)
    for f in self._get_layers(): model.add(f)


  def _get_layers(self):
    raise NotImplemented


  @staticmethod
  def parse_archi_string(archi_string: str):
    return [[int(s) for s in ss.split('-')]
            for ss in archi_string.split('=')]


  # region: Basic Layer Combinations

  @classmethod
  def conv_bn_relu(cls, filters, kernel_size, use_batchnorm=True, **kwargs):
    assert filters is not None
    return [Conv2D(filters, kernel_size, use_batchnorm=use_batchnorm,
                   activation='relu')]

  @classmethod
  def bottle_net(cls, filters, kernel_size, use_batchnorm=True):
    assert filters is not None
    layers = cls.conv_bn_relu(filters // 4, 1, use_batchnorm)
    layers.extend(cls.conv_bn_relu(filters // 4, kernel_size, use_batchnorm))
    layers.extend(cls.conv_bn_relu(filters, 1, use_batchnorm))
    return layers

  @classmethod
  def parse_layer_string(cls, spec, filters=None, use_batchnorm=True, **kwargs):
    assert isinstance(spec, str)
    spec = spec.lower()

    # Find layers
    if spec in ('conv1x1', 'c1'):
      return cls.conv_bn_relu(filters, 1, use_batchnorm, **kwargs)
    elif spec in ('conv3x3', 'c3'):
      return cls.conv_bn_relu(filters, 3, use_batchnorm, **kwargs)
    elif spec in ('maxpool3x3', 'm3'):
      return [MaxPool2D(pool_size=(3, 3), strides=1, **kwargs)]
    elif spec in ('bottlenetconv3x3', 'b3'):
      return cls.bottle_net(filters, 3, use_batchnorm, **kwargs)
    elif spec in ('bottlenetconv5x5', 'b5'):
      return cls.bottle_net(filters, 5, use_batchnorm, **kwargs)
    else: raise KeyError('!! Can not parse `{}`'.format(spec))

  # endregion: Basic Layer Combinations







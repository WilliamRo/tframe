from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

from tframe.utils import get_scale
from tframe.core.decorators import init_with_graph
from tframe.core.function import Function

from tensorflow.python.layers.pooling import MaxPooling1D as MaxPool1D_
from tensorflow.python.layers.pooling import MaxPool2D as MaxPool2D_
from tensorflow.python.layers.pooling import MaxPooling3D as MaxPool3D_
from tensorflow.python.layers.pooling import AveragePooling1D as AveragePooling1D_
from tensorflow.python.layers.pooling import AveragePooling2D as AveragePooling2D_

# TODO: this module needs to be refactored



class MaxPool1D(Layer, MaxPool1D_):

  full_name = 'maxpoold'
  abbreviation = 'maxpool'

  @init_with_graph
  def __init__(self, pool_size, strides,
               padding='same', data_format='channels_last',
               name=None, **kwargs):
    MaxPool1D_.__init__(
      self, pool_size, strides, padding, data_format, name, **kwargs)

  @property
  def structure_tail(self):
    size = lambda inputs: 'x'.join([str(n) for n in inputs])
    return '({}>{})'.format(size(self.pool_size), size(self.strides))

  @single_input
  def _link(self, input_=None, **kwargs):
    assert isinstance(input_, tf.Tensor)
    output = MaxPool1D_.__call__(self, input_, scope=self.full_name)
    # self.neuron_scale = get_scale(output)
    return output

  def __call__(self, *args, **kwargs):
    return Layer.__call__(self, *args, **kwargs)


class MaxPool2D(Layer, MaxPool2D_):

  full_name = 'maxpool2d'
  abbreviation = 'maxpool'

  @init_with_graph
  def __init__(self, pool_size, strides,
               padding='same', data_format='channels_last',
               name=None, **kwargs):
    MaxPool2D_.__init__(
      self, pool_size, strides, padding, data_format, name, **kwargs)

  @property
  def structure_tail(self):
    size = lambda inputs: 'x'.join([str(n) for n in inputs])
    return '({}>{})'.format(size(self.pool_size), size(self.strides))

  @single_input
  def _link(self, input_=None, **kwargs):
    assert isinstance(input_, tf.Tensor)
    output = MaxPool2D_.__call__(self, input_, scope=self.full_name)
    # self.neuron_scale = get_scale(output)
    return output

  def __call__(self, *args, **kwargs):
    return Layer.__call__(self, *args, **kwargs)


class MaxPool3D(Layer, MaxPool3D_):

  full_name = 'maxpool3d'
  abbreviation = 'maxpool'

  @init_with_graph
  def __init__(self, pool_size, strides,
               padding='same', data_format='channels_last',
               name=None, **kwargs):
    MaxPool3D_.__init__(
      self, pool_size, strides, padding, data_format, name, **kwargs)

  @property
  def structure_tail(self):
    size = lambda inputs: 'x'.join([str(n) for n in inputs])
    return '({}>{})'.format(size(self.pool_size), size(self.strides))

  @single_input
  def _link(self, input_=None, **kwargs):
    assert isinstance(input_, tf.Tensor)
    output = MaxPool3D_.__call__(self, input_, scope=self.full_name)
    # self.neuron_scale = get_scale(output)
    return output

  def __call__(self, *args, **kwargs):
    return Layer.__call__(self, *args, **kwargs)


class AveragePooling1D(Layer, AveragePooling1D_):

  full_name = 'avgpool1d'
  abbreviation = 'avgpool1d'

  @init_with_graph
  def __init__(self, pool_size, strides,
               padding='same', data_format='channels_last',
               name=None, **kwargs):
    AveragePooling1D_.__init__(
      self, pool_size, strides, padding, data_format, name, **kwargs)

  @property
  def structure_tail(self):
    size = lambda inputs: 'x'.join([str(n) for n in inputs])
    return '({}>{})'.format(size(self.pool_size), size(self.strides))

  @single_input
  def _link(self, input_=None, **kwargs):
    assert isinstance(input_, tf.Tensor)
    output = AveragePooling1D_.__call__(self, input_, scope=self.full_name)
    return output

  def __call__(self, *args, **kwargs):
    return Layer.__call__(self, *args, **kwargs)


class AveragePooling2D(Layer, AveragePooling2D_):

  full_name = 'avgpool2d'
  abbreviation = 'avgpool2d'

  @init_with_graph
  def __init__(self, pool_size, strides,
               padding='same', data_format='channels_last',
               name=None, **kwargs):
    AveragePooling2D_.__init__(
      self, pool_size, strides, padding, data_format, name, **kwargs)

  @property
  def structure_tail(self):
    size = lambda inputs: 'x'.join([str(n) for n in inputs])
    return '({}>{})'.format(size(self.pool_size), size(self.strides))

  @single_input
  def _link(self, input_=None, **kwargs):
    assert isinstance(input_, tf.Tensor)
    output = AveragePooling2D_.__call__(self, input_, scope=self.full_name)
    return output

  def __call__(self, *args, **kwargs):
    return Layer.__call__(self, *args, **kwargs)


class ReduceMean(Layer):

  full_name = 'average'
  abbreviation = 'avg'

  @init_with_graph
  def __init__(self, axis, keepdims=False, **kwargs):
    if not isinstance(axis, (list, tuple)): axis = [axis]

    self.axis = axis
    self.keepdims = keepdims

    self._kwargs = kwargs

  def get_layer_string(self, scale, full_name=False, suffix=''):
    suffix = f'({",".join([str(i) for i in self.axis])})'
    result = super().get_layer_string(scale, full_name, suffix)
    return result

  @single_input
  def _link(self, input_, **kwargs):
    output = tf.reduce_mean(input_, axis=self.axis, keepdims=self.keepdims)
    return output



class GlobalAveragePooling1D(Layer):

  full_name = 'globalavgpool1d'
  abbreviation = 'gap1d'

  @init_with_graph
  def __init__(self, data_format='channels_last', flatten=True, **kwargs):
    self._data_format = data_format
    assert data_format == 'channels_last'
    self._flatten = flatten
    self._kwargs = kwargs

  @single_input
  def _link(self, input_, **kwargs):
    assert isinstance(input_, tf.Tensor)
    shape = input_.shape.as_list()
    assert len(shape) == 3
    output = tf.layers.average_pooling1d(
      input_, pool_size=shape[1], strides=1, data_format=self._data_format)
    output = tf.reshape(output, shape=[-1, output.shape.as_list()[-1]])
    return output


class GlobalAveragePooling2D(Layer):

  full_name = 'globalavgpool2d'
  abbreviation = 'gap2d'

  @init_with_graph
  def __init__(self, data_format='channels_last', flatten=True, **kwargs):
    self._data_format = data_format
    assert data_format == 'channels_last'
    self._flatten = flatten
    self._kwargs = kwargs

  @single_input
  def _link(self, input_, **kwargs):
    assert isinstance(input_, tf.Tensor)
    shape = input_.shape.as_list()
    assert len(shape) == 4
    output = tf.layers.average_pooling2d(
      input_, pool_size=shape[1:3], strides=(1, 1),
      data_format=self._data_format)
    output = tf.reshape(output, shape=[-1, output.shape.as_list()[-1]])
    return output


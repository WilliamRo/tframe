from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import tensorflow as tf

from tframe.core.function import Function
from tframe import activations, initializers, checker, linker
from tframe.operators.apis.neurobase import NeuroBase


class Layer(Function):
  """Abstract definition for layers"""
  # If layer is nucleus, it will be wrapped as a sub net while added
  #  .. to a net
  is_nucleus = False

  # Full name will appear in tensor board
  full_name = None
  # Abbreviation will appear in structure string
  abbreviation = None

  # If not None, neuron scale will be shown in structure string
  neuron_scale = None
  output_scale = None

  @property
  def group_name(self):
    """group name will be shown in tensorboard graphs view.
       If full_name is not specified, no variable scope will be created"""
    return self.full_name

  @staticmethod
  def _get_variable(name, shape, fixed_zero=False,
                      initializer='xavier_uniform', regularizer=None):
    return tf.get_variable(
      name, shape, dtype=tf.float32, trainable=not fixed_zero,
      initializer=tf.zeros_initializer() if fixed_zero else initializer,
      regularizer=None if fixed_zero else regularizer)

  @property
  def structure_tail(self):
    if self.neuron_scale is not None:
      ns_str = 'x'.join(['{}'.format(d) for d in self.neuron_scale])
      return '({})'.format(ns_str)
    return ''

  def get_layer_string(self, scale, full_name=False, suffix=''):
    result = self.abbreviation if not full_name else self.full_name
    if scale: result += self.structure_tail
    result += suffix
    if self.output_id is not None:
      result += ':=' + self.output_id_str
    return result


def single_input(_link):

  def wrapper(*args, **kwargs):
    # Currently not sure if the decorator is for class method only
    input_ = args[1] if isinstance(args[0], Layer) else args[0]
    if isinstance(input_, (tuple, list)):
      if len(input_) != 1:
        raise ValueError('!! This layer only accept single input')
      input_ = input_[0]

    if input_ is not None and not isinstance(input_, tf.Tensor):
      raise TypeError('!! This layer only accept a Tensor as input')

    args = (args[0], input_) if isinstance(args[0], Layer) else (input_,)

    return _link(*args, *kwargs)

  return wrapper


class LayerWithNeurons(Layer, NeuroBase):
  is_nucleus = True

  def __init__(
      self,
      activation=None,
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      **kwargs):

    # Call parent's constructor
    NeuroBase.__init__(self, activation, weight_initializer,
                       use_bias, bias_initializer, **kwargs)

    # Common attributes
    self._activation_string = activation
    self._output_dim = None
    self.tensors_to_export = OrderedDict()

  @single_input
  def _link(self, x, **kwargs):
    return self.forward(x, **kwargs)

  def forward(self, x, **kwargs):
    raise NotImplemented

  def neurons(self,
              x,
              num=None,
              is_gate=False,
              activation=None,
              scope=None,
              truncate=False,
              num_or_size_splits=None,
              weight_initializer=None,
              use_bias=None,
              bias_initializer=None,
              weight_regularizer=None,
              bias_regularizer=None,
              activity_regularizer=None,
              prune_frac=0,
              **kwargs):
    if num is None:
      if isinstance(num_or_size_splits, int):
        assert self._output_dim is not None
        num = num_or_size_splits * self._output_dim
      elif isinstance(num_or_size_splits, (list, tuple)):
        num = sum(num_or_size_splits)
      else:
        assert self._output_dim is not None
        num = self._output_dim
    if activation is None and is_gate:
      activation = tf.sigmoid
    if weight_initializer is None:
      weight_initializer = getattr(self, '_weight_initializer', None)
    if use_bias is None:
      use_bias = getattr(self, '_use_bias', None)
    if bias_initializer is None:
      bias_initializer = getattr(self, '_bias_initializer')
    if weight_regularizer is None:
      weight_regularizer = getattr(self, '_weight_regularizer', None)
    if bias_regularizer is None:
      bias_regularizer = getattr(self, '_bias_regularizer', None)
    if activity_regularizer is None:
      activity_regularizer = getattr(self, '_activity_regularizer', None)
    return linker.neurons(
      num=num, external_input=x, activation=activation, scope=scope,
      use_bias=use_bias, truncate=truncate,
      num_or_size_splits=num_or_size_splits,
      weight_initializer=weight_initializer,
      bias_initializer=bias_initializer,
      weight_regularizer=weight_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      prune_frac=prune_frac,
      **kwargs)


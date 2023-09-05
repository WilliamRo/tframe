from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
from tframe import tf

import tframe as tfr
from tframe.utils.arg_parser import Parser
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

from tframe.utils.misc import get_scale

from tframe import activations
from tframe import hub
from tframe import initializers
from tframe import regularizers
from tframe import pedia


class Activation(Layer):
  """"""
  def __init__(self, identifier, set_logits=False):
    p = Parser.parse(identifier)
    self._id = p.name
    self.abbreviation = (p.name if isinstance(identifier, six.string_types)
                         else identifier.__name__)
    self.full_name = self.abbreviation
    self._activation = activations.get(identifier)
    self._set_logits = set_logits

  @single_input
  def _link(self, inputs, **kwargs):
    """Group name of Activation layer is decided not in calling
       Function.__call__ but calling self._activation"""
    outputs = self._activation(inputs)
    if self._id == 'softmax' or self._set_logits:
      tfr.context.set_logits_tensor(outputs, inputs)
    return outputs

  @staticmethod
  def ReLU():
    return Activation('relu')

  @staticmethod
  def LeakyReLU(leak):
    assert isinstance(leak, float) and leak > 0
    return Activation('lrelu:{}'.format(leak))


class Dropout(Layer):
  """"""
  abbreviation = 'dropout'
  full_name = abbreviation

  def __init__(self, train_keep_prob=0.5):
    # Initialize keep probability until while linking to put the
    #   the placeholder in the right name scope

    # self._keep_prob = None
    self.train_keep_prob = train_keep_prob

  @property
  def structure_tail(self):
    return '({:.2f})'.format(1 - self.train_keep_prob)

  @single_input
  def _link(self, input_, **kwargs):
    return tf.nn.dropout(
      input_, tf.cond(tf.get_collection(pedia.is_training)[0],
                      lambda: tf.constant(self.train_keep_prob, hub.dtype),
                      lambda: tf.constant(1.0, dtype=hub.dtype)))


class Linear(Layer):
  """Linear transformation layer, also known as fully connected layer or
     dense layer"""
  is_nucleus = True

  full_name = 'linear'
  abbreviation = 'fc'

  def __init__(self, output_dim,
               force_real=False,
               use_bias=True,
               weight_initializer='xavier_normal',
               bias_initializer='zeros',
               weight_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    if not np.isscalar(output_dim):
      raise TypeError('!! output_dim must be a scalar, not {}'.format(
        type(output_dim)))

    self._output_dim = output_dim
    self._force_real = force_real
    self._use_bias = use_bias

    self._weight_initializer = initializers.get(weight_initializer)
    self._bias_initializer = initializers.get(bias_initializer)
    self._weight_regularizer = regularizers.get(weight_regularizer, **kwargs)
    self._bias_regularizer = regularizers.get(bias_regularizer, **kwargs)

    self.weights = None
    self.biases = None

    self.neuron_scale = [output_dim]

  @single_input
  def _link(self, input_, **kwargs):
    assert isinstance(input_, tf.Tensor)

    # If this layer has been linked once, variables should be reused
    if self.weights is not None:
      tf.get_variable_scope().reuse_variables()

    # Get the shape and data type of input
    input_shape = input_.get_shape().as_list()
    dtype = input_.dtype

    weight_shape = (input_shape[-1], self._output_dim)
    bias_shape = (self._output_dim, )

    # Use lambda to make getting variable easier
    get_weight_variable = lambda name, fixed_zero=False: self._get_variable(
      name, weight_shape, fixed_zero, self._weight_initializer,
      self._weight_regularizer)
    get_bias_variable = lambda name, fixed_zero=False: self._get_variable(
      name, bias_shape, fixed_zero, self._bias_initializer,
      self._bias_regularizer)

    # Get variable
    if dtype in [tf.complex64, tf.complex128]:
      # Get complex weights and biases
      self.weights = tf.complex(
        get_weight_variable('weights_real'),
        get_weight_variable('weights_imag', self._force_real),
        name='weights')
      if self._use_bias:
        self.biases = tf.complex(
          get_bias_variable('biases_real'),
          get_bias_variable('biases_imag', self._force_real),
          name='biases')
    else:
      # Get real weights and biases
      self.weights = get_weight_variable('weights')
      if self._use_bias:
        self.biases = get_bias_variable('biases')

    # Calculate output
    output = tf.matmul(input_, self.weights)
    if self._use_bias:
      output += self.biases

    # Monitor
    # if hub.monitor_weight or hub.monitor_grad:
    #   tfr.monitor.add_weight(self.weights)

    self.output_tensor = output
    return output


class Rescale(Layer):
  full_name = 'rescale'
  abbreviation = 'rs'

  def __init__(self, from_scale, to_scale):
    if not(isinstance(from_scale, list) or isinstance(from_scale, tuple)):
      raise TypeError('from_scale must be a list or a tuple')
    if not(isinstance(to_scale, list) or isinstance(to_scale, tuple)):
      raise TypeError('to_scale must be a list or a tuple')

    self._from_scale = from_scale
    self._to_scale = to_scale

  @single_input
  def _link(self, input_, **kwargs):
    from_, to_ = self._from_scale, self._to_scale
    if from_[0] >= from_[1]:
      raise ValueError('from_[0] should be less than from_[1]')
    if to_[0] >= to_[1]:
      raise ValueError('to_[0] should be less than to_[1]')
    output = (input_ - from_[0]) / (from_[1] - from_[0])
    output = output * (to_[1] - to_[0]) + to_[0]

    return output


class Onehot(Layer):
  full_name = 'onehot'
  abbreviation = 'onehot'

  def __init__(self, depth):
    assert isinstance(depth, int) and depth > 1
    self._depth = depth
    self.neuron_scale = [depth]


  @single_input
  def _link(self, indices):
    assert isinstance(indices, tf.Tensor)
    shape = indices.shape.as_list()
    assert shape[-1] == 1
    return tf.one_hot(tf.squeeze(indices, axis=-1), self._depth)


class Reshape(Layer):

  def __init__(self, shape=None):
    """
    Reshape layer. 
    :param shape: list or tuple. Shape of each example, not including 1st
                   dimension
    """
    self.output_shape = shape
    name = 'flatten' if self.output_shape is None else 'reshape'
    self.abbreviation = name
    self.full_name = name

  @single_input
  def _link(self, input_, **kwargs):
    input_shape = input_.get_shape().as_list()
    output_shape = ([-1, np.prod(input_shape[1:])]
                    if self.output_shape is None
                    else [-1] + list(self.output_shape))

    output = tf.reshape(input_, output_shape, name=self.full_name)
    self.neuron_scale = get_scale(output)
    return output


class BatchReshape(Layer):
  """Say the input tensor has a shape of [bs, N * M], this layer reshapes
  this tensor to [bs * N, M]"""

  DEFAULT_PLACEHOLDER_KEY = 'tensor_block_size'

  def __init__(self, key=None):
    self.key = self.DEFAULT_PLACEHOLDER_KEY if key is None else key
    self.full_name = 'BatchReshape'
    self.abbreviation = self.full_name

  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    N = tf.placeholder(dtype=tf.int32, shape=(), name=self.key)
    # Add N to default collection
    tf.add_to_collection(pedia.default_feed_dict, N)

    x_shape = tf.shape(x)
    D = x_shape[1] // N

    # TODO: a workaround, '[-1, D] + x_shape[2:]' does not work
    output_shape = [-1, D] + [x_shape[i] for i in range(2, len(x.shape))]
    output = tf.reshape(x, output_shape, name=self.full_name)
    return output


class Input(Layer):

  def __init__(
      self,
      sample_shape=None,
      dtype=None,
      name='Input',
      group_shape=None):
    """Initialize an input layer.

    :param sample_shape: shape of each sample input to a model (at each time
                         step)
    :param dtype: data type of this input
    :param name: input name
    :param group_shape: [None] (represents batch dimension) by default. For
                        RNNs, it will be automatically set to [None, None]
                        in `recurrent.py -> Recurrent._build` method
    """
    # Check sample shape
    if sample_shape is not None:
      if not isinstance(sample_shape, (list, tuple)):
        raise TypeError('sample_shape must be a list or a tuple')

    # Initiate attributes
    self.sample_shape = sample_shape
    self.group_shape = None
    self.dtype = hub.dtype if dtype is None else dtype
    self.name = name
    self.place_holder = None
    self.rnn_single_step_input = None

    self.set_group_shape(group_shape)

    self.dummy = False


  @property
  def input_shape(self):
    if self.sample_shape is None: return self.sample_shape
    if self.group_shape is None: return [None] + list(self.sample_shape)
    else: return list(self.group_shape) + list(self.sample_shape)


  def set_group_shape(self, shape):
    """Currently, this method will be only called by Recurrent outside this
    module."""
    if shape is not None:
      if not isinstance(shape, (tuple, list)):
        raise TypeError('group_shape must be a list or a tuple')
    self.group_shape = shape


  def _link(self, *args, **kwargs):
    assert self.place_holder is None
    # This method is only accessible by Function.__call__ thus a None will
    #   be given as input
    assert len(args) == 0 and len(kwargs) == 0

    # Make dummy output if required
    if self.dummy:
      shape = [1] + list(self.sample_shape)
      return tf.constant(np.zeros(shape, dtype=float), dtype=self.dtype)

    input_ = tf.placeholder(
      dtype=self.dtype, shape=self.input_shape, name=self.name)

    # Update neuron scale
    self.neuron_scale = get_scale(input_)
    # Add input to default collection
    tf.add_to_collection(pedia.default_feed_dict, input_)
    # Return placeholder
    self.place_holder = input_

    # TODO: RNNs do not support partially unknown input shape
    # this block will cause an unnecessary reshape operator appearing
    # in graph
    if not None in self.sample_shape:
      self.rnn_single_step_input = tf.reshape(
        input_, [-1] + list(self.sample_shape))
    return input_


Flatten = lambda: Reshape()

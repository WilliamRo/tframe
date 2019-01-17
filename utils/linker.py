from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tframe.activations as activations
import tframe.initializers as initializers
import tframe.regularizers as regularizers

from tframe import hub
from tframe import context


# region : Standard units

def neurons(num,
            external_input,
            activation=None,
            memory=None,
            fc_memory=True,
            scope=None,
            use_bias=True,
            truncate=False,
            num_or_size_splits=None,
            weight_initializer='glorot_uniform',
            bias_initializer='zeros',
            weight_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            **kwargs):
  """Analogous to tf.keras.layers.Dense"""
  activation = activations.get(activation)
  weight_initializer = initializers.get(weight_initializer)
  bias_initializer = initializers.get(bias_initializer)
  weight_regularizer = regularizers.get(weight_regularizer)
  bias_regularizer = regularizers.get(bias_regularizer)
  activity_regularizer = regularizers.get(activity_regularizer)

  def forward():
    # Prepare a weight list for potential regularizer calculation
    weight_list = []
    x = (tf.concat([external_input, memory], axis=1, name='x_and_memory')
         if memory is not None and fc_memory else external_input)
    W = get_variable('W', [get_dimension(x), num], weight_initializer)
    weight_list.append(W)
    y = get_matmul(truncate)(x, W)
    if memory is not None and not fc_memory:
      memory_dim = get_dimension(memory)
      assert memory_dim == num
      Ws = get_variable('Ws', [1, num], weight_initializer)
      weight_list.append(Ws)
      y = tf.add(y, get_multiply(truncate)(memory, Ws))
    b = None
    if use_bias:
      b = get_bias('bias', num, bias_initializer)
      y = tf.nn.bias_add(y, b)
    if callable(activation): y = activation(y)
    return y, weight_list, b

  if scope is not None:
    with tf.variable_scope(scope): y, W_list, b = forward()
  else: y, W_list, b = forward()

  # Add regularizer if necessary
  if callable(weight_regularizer):
    context.add_loss_tensor(tf.add_n([weight_regularizer(w) for w in W_list]))
  if callable(bias_regularizer) and b is not None:
    context.add_loss_tensor(bias_regularizer(b))
  if callable(activity_regularizer):
    context.add_loss_tensor(activity_regularizer(y))

  if num_or_size_splits is not None:
    return tf.split(y, num_or_size_splits=num_or_size_splits, axis=1)
  return y

# endregion : Standard units

# region : Misc

def get_variable(name, shape, initializer='glorot_uniform'):
  initializer = initializers.get(initializer)
  return tf.get_variable(name, shape, dtype=hub.dtype, initializer=initializer)

def get_bias(name, dim, initializer='zeros'):
  initializer = initializers.get(initializer)
  return tf.get_variable(
    name, shape=[dim], dtype=hub.dtype, initializer=initializer)

def get_tensor_shape(tensor):
  assert isinstance(tensor, tf.Tensor)
  return tensor.shape.as_list()

def get_dimension(tensor):
  assert isinstance(tensor, tf.Tensor)
  shape_list = tensor.shape.as_list()
  assert len(shape_list) == 2
  return shape_list[1]

# endregion : Misc

# region : Operations

@tf.custom_gradient
def truncate_matmul(x, W):
  """Gradients do not back-prop through the first argument (a tensor)"""
  assert len(x.shape) == len(W.shape) == 2
  y = tf.matmul(x, W)
  def grad(dy):
    dx = tf.zeros_like(x)
    dW = tf.matmul(tf.transpose(x), dy)
    return dx, dW
  return y, grad

def get_matmul(truncate=False):
  if truncate: return truncate_matmul
  else: return tf.matmul

@tf.custom_gradient
def truncate_multiply(x, W):
  """Gradients do not back-prop through the first argument (a tensor)"""
  assert len(x.shape) == len(W.shape) == 2
  y = tf.multiply(x, W)
  def grad(dy):
    dx = tf.zeros_like(x)
    dW = tf.reduce_sum(tf.multiply(dy, x), axis=0, keepdims=True)
    return dx, dW
  return y, grad

def get_multiply(truncate=False):
  if truncate: return truncate_multiply
  else: return tf.multiply

# endregion : Operations

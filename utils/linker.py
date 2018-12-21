from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tframe.initializers as initializers
import tframe.regularizers as regularizers

from tframe import hub
from tframe import context


# region : Standard units

def neurons(num,
            x,
            activation=None,
            memory=None,
            fc_memory=True,
            scope=None,
            use_bias=True,
            truncate=False,
            weight_initializer='glorot_uniform',
            bias_initializer='zeros',
            weight_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            **kwargs):
  """Analogous to tf.keras.layers.Dense"""
  weight_initializer = initializers.get(weight_initializer)
  bias_initializer = initializers.get(bias_initializer)
  weight_regularizer = regularizers.get(weight_regularizer)
  bias_regularizer = regularizers.get(bias_regularizer)
  activity_regularizer = regularizers.get(activity_regularizer)

  def forward():
    weight_list = []
    Wx = get_variable('W', [get_dimension(x), num], weight_initializer)
    weight_list.append(Wx)
    y = get_matmul(truncate)(x, Wx)
    if memory is not None:
      assert isinstance(memory, tf.Tensor)
      memory_dim = get_dimension(memory)
      if fc_memory:
        op = get_matmul(truncate)
        Ws = get_variable('Ws', [memory_dim, num], weight_initializer)
      else:
        assert memory_dim == num
        op = get_multiply(truncate)
        Ws = get_variable('Ws', [1, num], weight_initializer)
      y = tf.add(y, op(memory, Ws))
      weight_list.append(Ws)
    b = get_bias('bias', num, bias_initializer) if use_bias else None
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

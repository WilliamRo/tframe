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
  if activation is not None: activation = activations.get(activation)
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
def truncate_multiply(x, y):
  """Gradients do not back-prop through the first argument (a tensor)"""
  assert len(x.shape) == len(y.shape) == 2
  z = tf.multiply(x, y)
  def grad(dz):
    dx = tf.zeros_like(x)
    dy = tf.multiply(dz, x)
    # TODO
    # db = tf.reduce_sum(tf.multiply(dy, a), axis=0, keepdims=True)
    return dx, dy
  return z, grad

def get_multiply(truncate=False):
  if truncate: return truncate_multiply
  else: return tf.multiply

# endregion : Operations

# region : Activations

def softmax_over_groups(net_input, configs, output_name='sog'):
  """ Ref: Grouped Distributor Unit (2019)
  """
  # Sanity check
  assert isinstance(net_input, tf.Tensor) and isinstance(configs, (list, tuple))
  for g in configs:
    assert isinstance(g, (tuple, list)) and len(g) in (2, 3)
    assert isinstance(g[0], int) and g[0] > 0
    assert isinstance(g[1], int) and g[1] > 0
    if len(g) == 3: assert 0 < g[2] <= g[0] or g[2] == -1
  group_sizes = [g[0] * g[1] for g in configs]
  assert sum(group_sizes) == get_dimension(net_input)

  # Calculate output
  splitted = (tf.split(net_input, group_sizes, axis=1) if len(group_sizes) > 0
              else [net_input])
  output_list = []
  # s: group size; n: group number
  # for (s, n), net_s in zip(groups, splitted):
  for g, net_s in zip(configs, splitted):
    d = None
    if len(g) == 2: s, n = g
    else: s, n, d = g
    activated = net_s
    if s == 1:
      if d == -1: activated = tf.sigmoid(activated)
      else: activated = tf.ones_like(activated)
    else:
      if n > 1: activated = tf.reshape(activated, [-1, s])
      activated = tf.nn.softmax(activated)
      if n > 1: activated = tf.reshape(activated, [-1, s*n])

    if d is not None:
      if 0 < d <= 1: activated = tf.multiply(d, activated)
      elif 1 < d <= s :
        # b for base
        b = 1.0 * (d - 1) / (s - 1)
        activated = tf.add(b, tf.multiply(1 - b, activated))
      else: assert d == -1

    output_list.append(activated)

  return (tf.concat(output_list, axis=1, name=output_name)
          if len(output_list) > 1 else output_list[0])

# endregion : Activations

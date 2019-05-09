from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tframe.activations as activations
import tframe.initializers as initializers
import tframe.regularizers as regularizers

from tframe import hub
from tframe import context
from tframe.utils.maths.periodicals import bit_waves


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
            allow_prune=False,
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

    if allow_prune and hub.pruning_rate_fc > 0.0:
      W = get_weights_to_prune('W', [get_dimension(x), num], weight_initializer)
    else: W = get_variable('W', [get_dimension(x), num], weight_initializer)

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
  v = tf.get_variable(name, shape, dtype=hub.dtype, initializer=initializer)
  return v

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
  """
  configs = ((size, num, [delta]), ...)
  Ref: Grouped Distributor Unit (2019)
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
      if 0 < d <= 1: activated = tf.multiply(float(d), activated)
      elif 1 < d <= s :
        # b for base
        b = 1.0 * (d - 1) / (s - 1)
        activated = tf.add(b, tf.multiply(1 - b, activated))
      else: assert d == -1

    output_list.append(activated)

  return (tf.concat(output_list, axis=1, name=output_name)
          if len(output_list) > 1 else output_list[0])

# endregion : Activations

# region : Bit Max

def _get_waves(num_bits):
  key = 'bit_waves_{}'.format(num_bits)
  if key in context.reuse_dict.keys(): return context.reuse_dict[key]
  waves = tf.constant(bit_waves(num_bits, stack=True, axis=-1), dtype=hub.dtype)
  context.reuse_dict[key] = waves
  return waves

def bit_max(x, num_classes, heads=1, sum_heads=False, **kwargs):
  """Bit max
  :param x: a tensor of shape (batch_size, dim)
  :param num_classes: output dimension
  :param heads: heads #
  :param sum_heads: whether to sum up along head dimension before outputting
  :return: a tensor y with the same shape as x, sum(y[k, :]) == 1 for all k
  """
  # Sanity check
  assert isinstance(num_classes, int) and num_classes > 1
  assert isinstance(heads, int) and heads > 0
  # Get bits
  num_bits = int(np.ceil(np.log2(num_classes)))

  # Calculate activations for bits
  # a.shape = (bs, num_bits*heads)
  a = neurons(
    num_bits*heads, x, activation='sigmoid', scope='bit_activation', **kwargs)
  if heads > 1:
    # a.shape => (heads, bs, num_bits)
    a = tf.reshape(a, [-1, num_bits, heads])
    a = tf.transpose(a, [2, 0, 1])
  # a.shape => ([heads, ]bs, 2**num_bits, num_bits)
  a = tf.stack([a] * (2 ** num_bits), axis=-2)

  # Calculate bit_max
  # waves.shape = (1, 2**num_bits, num_bits)
  waves = _get_waves(num_bits)
  coef = tf.subtract(1., tf.multiply(2., a))
  wave_stack = tf.add(tf.multiply(waves, coef), a)
  # bit_max.shape = ([heads, ]bs, 2**num_bits)
  bit_max = tf.reduce_prod(wave_stack, axis=-1)

  # Trim if necessary
  scale = 2 ** num_bits
  if scale > num_classes:
    delta = (scale - num_classes) // 2
    # Trim
    if heads == 1: bit_max = bit_max[:, delta:num_classes+delta]
    else: bit_max = bit_max[:, :, delta:num_classes+delta]
    # Normalize
    sums = tf.reduce_sum(bit_max, axis=-1, keepdims=True) + 1e-6
    bit_max = tf.divide(bit_max, sums)

  # Add up if necessary
  if heads > 1 and sum_heads:
    bit_max = tf.reduce_sum(bit_max, axis=0)

  return bit_max

# endregion : Bit Max

# region : Prune

def get_weights_to_prune(name, shape, initializer):
  """Get variable and register this variable to context
  """
  # Get variable
  weights = get_variable(name, shape, initializer)
  # Register, context.pruner should be created in early model.build
  assert context.pruner is not None
  masked_weights = context.pruner.register_to_dense(weights)
  # Return
  assert isinstance(masked_weights, tf.Tensor)
  return masked_weights

# endregion : Prune


if __name__ == '__main__':
  a = 12
  print(int(np.ceil(np.log2(a))))


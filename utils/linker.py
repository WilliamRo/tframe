from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import tf
import tframe.activations as activations
import tframe.initializers as initializers
import tframe.regularizers as regularizers

from tframe import hub
from tframe import checker
from tframe import context
from tframe import pedia
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
            **kwargs):
  """Analogous to tf.keras.layers.Dense"""
  # Get activation, initializers and regularizers
  if activation is not None: activation = activations.get(activation)
  weight_initializer = initializers.get(weight_initializer)
  bias_initializer = initializers.get(bias_initializer)
  weight_regularizer = regularizers.get(weight_regularizer)
  bias_regularizer = regularizers.get(bias_regularizer)
  activity_regularizer = regularizers.get(activity_regularizer)

  # a. Check prune configs
  if 'prune_frac' in kwargs.keys():
    x_prune_frac, s_prune_frac = (kwargs['prune_frac'],) * 2
  else:
    x_prune_frac = kwargs.get('x_prune_frac', 0)
    s_prune_frac = kwargs.get('s_prune_frac', 0)
  prune_is_on = (isinstance(hub.pruning_rate, float)
                 and hub.pruning_rate > 0.0
                 and x_prune_frac + s_prune_frac > 0)

  # b. Check sparse configs
  x_heads = kwargs.get('x_heads', 0)
  s_heads = kwargs.get('s_heads', 0)
  sparse_is_on = x_heads + s_heads > 0

  # :: Decide to concatenate or not considering a and b
  # .. a
  if memory is None: should_concate = False
  elif prune_is_on: should_concate = x_prune_frac == s_prune_frac
  else: should_concate = fc_memory
  # .. b
  should_concate = should_concate and not sparse_is_on
  #
  separate_memory_neurons = memory is not None and not should_concate

  def get_weights(name, tensor, p_frac, heads):
    shape = [get_dimension(tensor), num]
    if prune_is_on and p_frac > 0:
      assert heads == 0
      return get_weights_to_prune(name, shape, weight_initializer, p_frac)
    elif heads > 0:
      return _get_sparse_weights(shape[0], shape[1], heads, use_bit_max=True,
                                 coef_initializer=weight_initializer)
    else: return get_variable(name, shape, weight_initializer)

  def forward():
    # Prepare a weight list for potential regularizer calculation
    weight_list = []

    # Get x
    x = (tf.concat([external_input, memory], axis=1, name='x_concat_s')
         if should_concate else external_input)

    # - Calculate net input for x
    # .. get weights
    name = 'Wx' if separate_memory_neurons else 'W'
    Wx = get_weights(name, x, x_prune_frac, x_heads)
    weight_list.append(Wx)
    # .. append weights to context, currently only some extractors will use it
    context.weights_list.append(Wx)
    # .. do matrix multiplication
    net_y = get_matmul(truncate)(x, Wx)

    # - Calculate net input for memory and add to net_y if necessary
    if separate_memory_neurons:
      if not fc_memory:
        assert not (prune_is_on and s_prune_frac > 0)
        memory_dim = get_dimension(memory)
        assert memory_dim == num
        Ws = get_variable('Ws', [1, num], weight_initializer)
        net_s = get_multiply(truncate)(memory, Ws)
      else:
        assert prune_is_on or sparse_is_on
        Ws = get_weights('Ws', memory, s_prune_frac, s_heads)
        net_s = get_matmul(truncate)(memory, Ws)

      # Append Ws to weight list and add net_s to net_y
      weight_list.append(Ws)
      net_y = tf.add(net_y, net_s)

    # - Add bias if necessary
    b = None
    if use_bias:
      b = get_bias('bias', num, bias_initializer)
      net_y = tf.nn.bias_add(net_y, b)

    # - Activate and return
    if callable(activation): net_y = activation(net_y)
    return net_y, weight_list, b

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

  # Split if necessary
  if num_or_size_splits is not None:
    return tf.split(y, num_or_size_splits=num_or_size_splits, axis=1)
  return y

# endregion : Standard units

# region : Unit with mask

def masked_neurons(x,
                   num,
                   scope,
                   activation=None,
                   s=None,
                   x_mask=None,
                   s_mask=None,
                   use_bias=True,
                   weight_initializer='glorot_normal',
                   bias_initializer='zeros',
                   **kwargs):
  # Sanity check
  assert isinstance(x, tf.Tensor)

  # Get activation and initializers
  if activation is not None: activation = activations.get(activation)
  weight_initializer = initializers.get(weight_initializer)
  bias_initializer = initializers.get(bias_initializer)

  def matmul(x, y):
    batch_matmul = len(x.shape) == len(y.shape) - 1
    if batch_matmul: x = tf.expand_dims(x, axis=1)
    assert len(x.shape) == len(y.shape)
    output = tf.matmul(x, y)
    if batch_matmul: output = tf.squeeze(output, axis=1)
    return output

  def get_weights(tensor, name, mask=None):
    shape = [get_dimension(tensor), num]
    if mask is None: return get_variable(name, shape, weight_initializer)
    else: return get_masked_weights(name, shape, weight_initializer, mask)

  def forward():
    # x -> y
    Wx = get_weights(x, 'Wx', x_mask)
    # .. do matrix multiplication
    net_y = matmul(x, Wx)
    # s -> y if s exists
    if s is not None:
      assert isinstance(s, tf.Tensor)
      Ws = get_weights(s, 'Ws', s_mask)
      # .. add s * Ws to net_y
      net_y = tf.add(net_y, matmul(s, Ws))
    # Add bias if necessary
    if use_bias:
      b = get_bias('bias', num, bias_initializer)
      net_y = tf.nn.bias_add(net_y, b)
    # Activate if necessary
    if activation is not None: return activation(net_y)
    else: return net_y

  with tf.variable_scope(scope): y = forward()

  # Return
  return y

# endregion : Unit with mask

# region : Utils

def get_variable(name, shape, initializer='glorot_uniform'):
  initializer = initializers.get(initializer)
  v = tf.get_variable(name, shape, dtype=hub.dtype, initializer=initializer)
  return v

def get_bias(name, dim, initializer='zeros'):
  initializer = initializers.get(initializer)
  return tf.get_variable(
    name, shape=[dim], dtype=hub.dtype, initializer=initializer)

def get_tensor_shape(tensor):
  assert isinstance(tensor, (tf.Tensor, tf.Variable))
  return tensor.shape.as_list()

def get_dimension(tensor):
  assert isinstance(tensor, tf.Tensor)
  shape_list = tensor.shape.as_list()
  # assert len(shape_list) == 2
  return shape_list[-1]

# endregion : Utils

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
  splitted = split(net_input, configs)
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
  waves = tf.constant(bit_waves(
    num_bits, stack=True, stack_axis=-1), dtype=hub.dtype)
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

  # output shape is ([heads, ]bs, num_classes)
  return bit_max

def expand_bit(tensor, axis, activate=True):
  """Given axis k, expand a tensor x of shape [d0, d1, ..., dk, ..., dn] to
     y of shape [d0, d1, ..., 2^dk, ..., dn] which satisfies
     all(reduce_sum(y, axis=k) == 1) is True
  """
  # Sanity check
  assert isinstance(tensor, (tf.Tensor, tf.Variable))
  assert isinstance(axis, int) and axis < len(tensor.shape)
  shape = get_tensor_shape(tensor)
  n = shape[axis]

  # Activate tensor if necessary
  # TODO: temporally borrow hub.multiple as the multiplier
  if activate: tensor = tf.sigmoid(tensor * hub.sigmoid_coef)
  # Expand tensor to shape [d0, ..., dk-1,   1, n, ..., dn]
  a = tf.expand_dims(tensor, axis=axis)

  # Get waves of shape (2^n, n)
  waves = _get_waves(n)
  # Reshape waves to       [ 1, ...,    1, 2^n, n, ...,  1]
  new_shape = [1] * (len(shape) + 1)
  new_shape[axis:axis+2] = (2**n, n)
  waves = tf.reshape(waves, new_shape)

  # Calculate bit max
  coef = tf.subtract(1., tf.multiply(2., a))
  y = tf.add(tf.multiply(coef, waves), a)
  y = tf.reduce_prod(y, axis=axis+1)
  return y

def _trim_and_normalize(tensor, axis, dim, normalize=False):
  # Sanity check
  assert isinstance(tensor, (tf.Tensor, tf.Variable))
  shape = tensor.shape.as_list()
  assert isinstance(axis, int) and axis < len(shape)
  scale = shape[axis]
  assert scale > dim

  # Trim
  begin, size = [0] * len(shape), shape
  delta = (scale - dim) // 2
  begin[axis], size[axis] = delta, dim
  output = tf.slice(tensor, begin, size)
  assert isinstance(output, (tf.Tensor, tf.Variable))
  assert output.shape.as_list()[axis] == dim

  # Normalize if necessary
  if normalize:
    sums = tf.reduce_sum(output, axis=axis, keepdims=True) + 1e-6
    output = tf.divide(output, sums)

  return output

# endregion : Bit Max

# region : Prune

def get_weights_to_prune(name, shape, initializer, frac):
  """Get variable and register this variable to context
  """
  # Get variable
  weights = get_variable(name, shape, initializer)
  # Register, context.pruner should be created in early model.build
  assert context.pruner is not None
  etch_config = 'lottery:prune_frac={}'.format(frac)
  masked_weights = context.pruner.register_to_kernels(weights, etch_config)
  # Return
  assert isinstance(masked_weights, tf.Tensor)
  return masked_weights

def get_masked_weights(name, shape, initializer, mask):
  """Dynamic weights are not to be registered into pruner"""
  # Get variable
  weights = get_variable(name, shape, initializer)
  # Register weights with mask if mask is static
  is_static = len(mask.shape) == 2
  if is_static:
    assert context.pruner is not None
    masked_weights = context.pruner.register_with_mask(weights, mask)
  else:
    masked_weights = weights * mask
  # Return
  assert isinstance(masked_weights, tf.Tensor)
  return masked_weights

# endregion : Prune

# region : Hyper affine

def hyper_affine(x, dim, heads=1, use_bit_max=True):
  """Hyper affine transformation.

  :param x: input tensor, must be of shape (batch_size, dim)
  :param dim: output dimension
  :param heads: head # for each output neuron
  :param use_bit_max: whether to use bix_max to calculate transformation matrix
  """
  # Sanity check, x.shape = (batch_size, x_dim)
  assert isinstance(x, tf.Tensor) and len(x.shape) == 2

  # Generate transformation matrix
  if use_bit_max:
    pass
  else: raise NotImplementedError

  # Do matrix multiplication
  y = None

  return y

# endregion : Hyper affine

# region : Sparse affine

def _get_sparse_weights(x_dim, y_dim, heads=1, use_bit_max=False,
                        logits_initializer='random_normal',
                        coef_initializer='random_normal',
                        return_package=False):

  logits_initializer = initializers.get(logits_initializer)
  coef_initializer = initializers.get(coef_initializer)

  # Get 3-D variable of shape (x_dim, y_dim, heads)
  if use_bit_max:
    num_bits = int(np.ceil(np.log2(x_dim)))
    logits = tf.get_variable(
      'brick', shape=[num_bits, y_dim, heads], dtype=hub.dtype,
      initializer=logits_initializer, trainable=True)
    activation = expand_bit(logits, axis=0)
    # Trim if necessary
    if 2**num_bits > x_dim:
      activation = _trim_and_normalize(
        activation, axis=0, dim=x_dim, normalize=True)
  else:
    logits = tf.get_variable(
      'logits', shape=[x_dim, y_dim, heads], dtype=hub.dtype,
      initializer=logits_initializer, trainable=True)
    activation = tf.nn.softmax(logits, axis=0)

  # Get coef variable of shape (y_dim, heads)
  coef_shape = [x_dim, y_dim, 1] if hub.full_weight else [1, y_dim, heads]
  coef = tf.get_variable('coef', shape=coef_shape, dtype=hub.dtype,
                         initializer=coef_initializer, trainable=True)

  # Calculate weight matrix
  weights = tf.reduce_sum(tf.multiply(coef, activation), axis=-1)
  assert weights.shape.as_list() == [x_dim, y_dim]
  context.sparse_weights_list.append(weights)

  # Return
  if return_package:
    package = {
      'logits': logits,
      'activation': activation,
      'coef': coef,
    }
    return weights, package
  else: return weights

def sparse_affine(x, y_dim, heads=1, use_bit_max=False,
                  logits_initializer='random_normal',
                  coef_initializer='random_normal', use_bias=True,
                  bias_initializer='zeros', return_package=False):

  """This method should be used inside a variable scope"""
  bias_initializer = initializers.get(bias_initializer)

  # Sanity check
  assert isinstance(x, tf.Tensor) and len(x.shape) == 2
  x_dim = get_dimension(x)

  # Get sparse weights
  weights, package = _get_sparse_weights(
    x_dim, y_dim, heads, use_bit_max,
    logits_initializer, coef_initializer, True)
  assert weights.shape.as_list() == [x_dim, y_dim]

  # Calculate y
  y = tf.matmul(x, weights)
  bias = get_bias('bias', y_dim, bias_initializer) if use_bias else None
  y = tf.nn.bias_add(y, bias)

  # Return
  if return_package:
    package['weights'] = weights
    return y, package
  else: return y

# endregion : Sparse affine

# region : MISC

def split_by_sizes(tensor_batch, sizes):
  assert isinstance(sizes, (tuple, list))
  if len(sizes) == 1: return [tensor_batch]
  return tf.split(tensor_batch, sizes, axis=1)

def split(tensor_batch, groups):
  assert isinstance(groups, (list, tuple)) and len(groups) >= 1
  groups = [g[:2] for g in groups]
  group_sizes = [s*n for s, n in groups]
  return split_by_sizes(tensor_batch, group_sizes)

def concatenate(tensor_list):
  assert isinstance(tensor_list, list) and len(tensor_list) > 0
  if len(tensor_list) == 1: return tensor_list[0]
  else: return tf.concat(tensor_list, axis=1)

def dropout(input_, dropout_rate, rescale=True):
  keep_prob = 1 - dropout_rate
  assert 0 < keep_prob < 1
  p = tf.cond(tf.get_collection(pedia.is_training)[0],
              lambda: keep_prob, lambda: 1.0)
  output = tf.nn.dropout(input_, keep_prob=p)
  if not rescale: return output * p
  return output

# endregion : MISC



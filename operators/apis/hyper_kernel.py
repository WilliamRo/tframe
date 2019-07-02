from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import hub

from .neurobase import RNeuroBase


class HyperKernel(RNeuroBase):
  """Provide kernels for hyper RNNs"""

  def _get_hyper_kernel(self, kernel_key, **kwargs):
    """In the future, initializers in kernel may be different from that in
       main cells. Use kwargs in this circumstances.
    """
    assert len(kwargs) == 0

    if kernel_key in ('rnn', 'srn', 'vanilla'): kernel = self._srn
    elif kernel_key == 'gru': kernel = self._gru
    elif kernel_key == 'gruv3': kernel = self._gruv3
    elif kernel_key == 'dcgru': kernel = self._dcgru
    elif kernel_key == 'gru4g': kernel = self._gru4g
    elif kernel_key in ['ugrnn']: kernel = self._ugrnn
    elif kernel_key in ['lstm']: kernel = self._lstm
    elif kernel_key in ['cplstm']: kernel = self._cplstm
    elif kernel_key in ['cm', 'column_mask']: kernel = self._column_mask
    else: raise KeyError('!! Unknown hyper key `{}`'.format(kernel_key))

    def kernel_with_scope(x, s):
      with tf.variable_scope('hyper_' + kernel_key):
        return kernel(x, s)

    return kernel_with_scope


  @staticmethod
  def _get_hyper_state_holder(key, size):
    get_holder = lambda name: tf.placeholder(hub.dtype, [None, size], name)
    if key in ['lstm', 'cplstm']:
      return get_holder('hyper_h'), get_holder('hyper_c')
    return get_holder('hyper_s')


  def _srn(self, x, prev_s):
    y = self.dense_rn(x, prev_s, 'hyper_srn', 'tanh')
    return y, y

  def _gru(self, x, prev_s):
    state_size = self.get_state_size(prev_s)
    r, z = self.dense_rn(x, prev_s, 'gate_block', is_gate=True,
                         output_dim=2*state_size, num_or_size_splits=2)
    s_bar = self.dense_rn(x, r*prev_s, scope='s_bar', activation='tanh')
    new_s = z * prev_s + (1. - z) * s_bar
    return new_s, new_s

  def _gruv3(self, x, prev_s):
    state_size = self.get_state_size(prev_s)
    r = self.dense_rn(x, prev_s, 'r_gate', is_gate=True, output_dim=state_size)
    s = r * prev_s
    s_bar, z = self.dense_rn(x, s, 'net_block', output_dim=2*state_size,
                             num_or_size_splits=2)
    s_bar, z = tf.tanh(s_bar), tf.sigmoid(z)
    new_s = z * prev_s + (1. - z) * s_bar
    return new_s, new_s

  def _dcgru(self, x, prev_s):
    state_size = self.get_state_size(prev_s)
    s = tf.tanh(prev_s)
    r, u, z = self.dense_rn(x, s, 'gate_block', is_gate=True,
                            output_dim=3*state_size, num_or_size_splits=3)
    s_bar = self.dense_rn(x, r*s, scope='s_bar', activation='tanh')
    new_s = z * prev_s + u * s_bar
    return tf.tanh(new_s), new_s

  def _gru4g(self, x, c):
    state_size = self.get_state_size(c)
    h = tf.tanh(c)
    h = self.dense_rn(x, h, 'r_gate', is_gate=True, output_dim=state_size) * h
    u, z, o, g = self.dense_rn(x, h, 'gate_block', output_dim=4*state_size,
                               num_or_size_splits=4)
    new_c = tf.sigmoid(z) * c + tf.sigmoid(u) * tf.tanh(g)
    y = tf.sigmoid(o) * tf.tanh(new_c)
    return y, new_c

  def _ugrnn(self, x, prev_s):
    state_size = self.get_state_size(prev_s)
    a_s_bar, a_z = self.dense_rn(
      x, prev_s, 'gate_block', output_dim=2 * state_size,
      num_or_size_splits=2)
    s_bar, z = tf.tanh(a_s_bar), tf.sigmoid(a_z)
    new_s = z * prev_s + (1. - z) * s_bar
    return new_s, new_s

  def _lstm(self, x, prev_s):
    h, c = prev_s
    dim = self.get_state_size(h)

    f, i, o, g = self.dense_rn(x, h, 'fiog', output_dim=dim*4,
                               num_or_size_splits=4)
    sigma, tanh = tf.sigmoid, tf.tanh
    new_c = sigma(i) * tanh(g) + sigma(f) * c
    new_h = sigma(o) * tanh(new_c)
    return new_h, (new_h, new_c)

  def _cplstm(self, x, prev_s):
    h, c = prev_s
    dim = self.get_state_size(h)

    f, o, g = self.dense_rn(
      x, h, 'fiog', output_dim=dim*3, num_or_size_splits=3)
    sigma, tanh = tf.sigmoid, tf.tanh
    f = sigma(f)
    new_c = (1. - f) * tanh(g) + f * c
    new_h = sigma(o) * tanh(new_c)
    return new_h, (new_h, new_c)

  def _column_mask(self, x, prev_s):
    x_dim = self.get_dimension(x)
    state_size = self.get_state_size(prev_s)
    rx = self.dense_rn(x, prev_s, 'rx', is_gate=True, output_dim=x_dim)
    rs, z =  self.dense_rn(x, prev_s, 'gate_block', is_gate=True,
                           output_dim=state_size * 2, num_or_size_splits=2)
    s_bar = self.dense_rn(rx*x, rs*prev_s, 's_bar', 'tanh')
    new_s = z * prev_s + (1. - z) * s_bar
    return new_s, new_s

  def _get_embeddings_hyper16(self, s_hat, num_cluster, signal_size):
    """Get num_cluster groups of embedding vectors.
       For each cluster, 3 signals should be generated:

         z_s = W_s @ s_hat + b_s
         z_x = W_x @ s_hat + b_x
         z_b = W_b @ s_hat

       Variable               Initializer
       W_s and W_x            zero
       b_s and b_x            one
       W_b                    normal(std=0.01)

     Ref: [1] Ha, etc. Hyper Neural Networks. 2016

    :param num_cluster: neuron groups number, e.g., LSTM has 4 groups
    :param signal_size: dimension of each embedding vector
    :return: ((zx1, zs1, zb1), (zx2, zs2, zb2), ...)
    """
    state_size = self.get_state_size(s_hat)

    # Get weight shape
    nxs = num_cluster * signal_size
    shape = [state_size, nxs * 3]
    # Create weights initializer according to [1]
    np_w_init = np.zeros(shape=shape, dtype=np.float32)
    np_w_init[:, nxs * 2:] = np.random.randn(state_size, nxs) * 0.01
    initializer = tf.initializers.constant(np_w_init)

    # Get weights and do matrix multiplication
    W = tf.get_variable(
      'Wh_hat', shape=shape, dtype=hub.dtype, initializer=initializer)
    z_block = s_hat @ W

    # Add bias
    bias = tf.get_variable('z_bias', shape=[nxs * 2],
                           dtype=hub.dtype, initializer=tf.initializers.ones)
    pad = tf.zeros(shape=[nxs], dtype=hub.dtype)
    bias = tf.concat([bias, pad], axis=0)

    z_block = tf.nn.bias_add(z_block, bias)
    zs = tf.split(z_block, 3 * num_cluster, axis=1)

    embeddings = []
    for _ in range(num_cluster):
      embeddings.append([zs.pop(0), zs.pop(0), zs.pop(-1)])
    return embeddings


  def _hyper_neuron_16(self, x, s, signals, scope, activation=None,
                       output_dim=None, is_gate=False):
    # Sanity check
    assert isinstance(signals, (list, tuple)) and len(signals) == 3
    signal_size = self.get_dimension(signals[0])
    if output_dim is None: output_dim = self.get_state_size(s)

    # Create a neuron array
    na = self.differentiate(
      output_dim, scope, activation, use_bias=True,
      weight_initializer=tf.initializers.orthogonal,
      bias_initializer='zeros', is_gate=is_gate)

    psi_const = 0.1 / signal_size
    na.add_kernel(x, 'hyper16', suffix='x', seed=signals[0],
                  seed_weight_initializer=psi_const)
    na.add_kernel(s, 'hyper16', suffix='s', seed=signals[1],
                  seed_weight_initializer=psi_const)
    na.register_bias_kernel(
      'hyper16', seed=signals[2], weight_initializer='zeros')

    return na()

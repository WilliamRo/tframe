from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe import hub

from .neurobase import RNeuroBase


class HyperKernel(RNeuroBase):
  """Provide kernels for hyper RNNs"""

  def _get_hyper_kernel(self, kernel_key, do=0, zo=0, ln=False, **kwargs):
    """In the future, initializers in kernel may be different from that in
       main cells. Use kwargs in this circumstances.

       If ln is True, Layer Normalization will be applied to each cell unit
    """
    # assert len(kwargs) == 0
    self._hdo = do
    self._hzo = zo
    self._hln = ln
    self._hyper_kwargs = kwargs

    if kernel_key in ('rnn', 'srn', 'vanilla'): kernel = self._srn
    elif kernel_key == 'gru': kernel = self._gru
    elif kernel_key == 'gruv3': kernel = self._gruv3
    elif kernel_key == 'dcgru': kernel = self._dcgru
    elif kernel_key == 'dcgruv2': kernel = self._dcgruv2
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

  def _update_states(self, z, prev_s, u, s_bar):
    if self._hdo > 0: s_bar = self.dropout(s_bar, self._hdo)
    return z * prev_s + u * s_bar

  def _srn(self, x, prev_s):
    y = self._dense_h(x, prev_s, 'hyper_srn', 'tanh')
    return y, y

  def _gru(self, x, prev_s):
    state_size = self.get_state_size(prev_s)
    r, z = self._dense_h(x, prev_s, 'gate_block', is_gate=True,
                         output_dim=2*state_size, num_or_size_splits=2)
    s_bar = self._dense_h(x, r*prev_s, scope='s_bar', activation='tanh')
    new_s = self._update_states(z, prev_s, 1. - z, s_bar)
    return new_s, new_s

  def _gruv3(self, x, prev_s):
    state_size = self.get_state_size(prev_s)
    r = self._dense_h(x, prev_s, 'r_gate', is_gate=True, output_dim=state_size)
    s = r * prev_s
    s_bar, z = self._dense_h(x, s, 'net_block', output_dim=2*state_size,
                             num_or_size_splits=2)
    s_bar, z = tf.tanh(s_bar), tf.sigmoid(z)
    new_s = self._update_states(z, prev_s, 1. - z, s_bar)
    return new_s, new_s

  def _dcgru(self, x, prev_s):
    state_size = self.get_state_size(prev_s)
    s = tf.tanh(prev_s)
    r, u, z = self._dense_h(x, s, 'gate_block', is_gate=True,
                            output_dim=3*state_size, num_or_size_splits=3)
    s_bar = self._dense_h(x, r*s, scope='s_bar', activation='tanh')
    new_s = self._update_states(z, prev_s, u, s_bar)
    return tf.tanh(new_s), new_s

  def _dcgruv2(self, x, prev_s):
    sigma, tanh = tf.sigmoid, tf.tanh
    state_size = self.get_state_size(prev_s)
    s = tanh(prev_s)
    r = self._dense_h(x, s, 'reset_gate', is_gate=True, output_dim=state_size)
    s = r * s
    u, z, s_bar = self._dense_h(
      x, s, 'gate_block', output_dim=3*state_size, num_or_size_splits=3)
    new_s = self._update_states(sigma(z), prev_s, sigma(u), tanh(s_bar))
    return tanh(new_s), new_s

  def _gru4g(self, x, c):
    state_size = self.get_state_size(c)
    h = tf.tanh(c)
    h = self._dense_h(x, h, 'r_gate', is_gate=True, output_dim=state_size) * h
    u, z, o, g = self._dense_h(x, h, 'gate_block', output_dim=4*state_size,
                               num_or_size_splits=4)
    new_c = self._update_states(tf.sigmoid(z), c, tf.sigmoid(u), tf.tanh(g))
    y = tf.sigmoid(o) * tf.tanh(new_c)
    return y, new_c

  def _ugrnn(self, x, prev_s):
    state_size = self.get_state_size(prev_s)
    forget_bias = self._hyper_kwargs.get('forget_bias', 0)
    if forget_bias == 0:
      a_s_bar, a_z = self._dense_h(
        x, prev_s, 'gate_block', output_dim=2 * state_size,
        num_or_size_splits=2)
    else:
      a_z = self._dense_h(x, prev_s, 'net_z', output_dim=state_size,
                          bias_initializer=forget_bias)
      a_s_bar = self._dense_h(x, prev_s, 'net_s_bar', output_dim=state_size)
    s_bar, z = tf.tanh(a_s_bar), tf.sigmoid(a_z)
    new_s = self._update_states(z, prev_s, 1. - z, s_bar)
    return new_s, new_s

  def _lstm(self, x, prev_s):
    forget_bias = self._hyper_kwargs.get('forget_bias', 0)
    h, c = prev_s
    dim = self.get_state_size(h)

    if forget_bias == 0:
      f, i, o, g = self._dense_h(
        x, h, 'fiog', output_dim=dim*4, num_or_size_splits=4)
    else:
      f = self._dense_h(
        x, h, 'net_f', bias_initializer=forget_bias, output_dim=dim)
      i, o, g = self._dense_h(
        x, h, 'iog', output_dim=dim*3, num_or_size_splits=3)
    sigma, tanh = tf.sigmoid, tf.tanh
    new_c = self._update_states(sigma(f), c, sigma(i), tanh(g))
    new_h = sigma(o) * tanh(new_c)
    return new_h, (new_h, new_c)

  def _cplstm(self, x, prev_s):
    h, c = prev_s
    dim = self.get_state_size(h)

    f, o, g = self._dense_h(
      x, h, 'fiog', output_dim=dim*3, num_or_size_splits=3)
    sigma, tanh = tf.sigmoid, tf.tanh
    f = sigma(f)
    new_c = self._update_states(f, c, 1. - f, tanh(g))
    new_h = sigma(o) * new_c
    return new_h, (new_h, new_c)

  def _column_mask(self, x, prev_s):
    x_dim = self.get_dimension(x)
    state_size = self.get_state_size(prev_s)
    rx = self._dense_h(x, prev_s, 'rx', is_gate=True, output_dim=x_dim)
    rs, z =  self._dense_h(x, prev_s, 'gate_block', is_gate=True,
                           output_dim=state_size * 2, num_or_size_splits=2)
    s_bar = self._dense_h(rx*x, rs*prev_s, 's_bar', 'tanh')
    new_s = self._update_states(z, prev_s, 1. - z, s_bar)
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


  def _dense_h(self, x, s, scope, activation=None, output_dim=None,
               is_gate=False, num_or_size_splits=None, **kwargs):
    if not self._hln or num_or_size_splits is None:
      return self.dense_rn(
        x, s, scope, activation=activation, output_dim=output_dim,
        is_gate=is_gate, num_or_size_splits=num_or_size_splits,
        layer_normalization=self._hln, **kwargs)
    # Get sizes
    if isinstance(num_or_size_splits, int):
      sizes = [int(output_dim/num_or_size_splits)] * num_or_size_splits
    else: sizes = num_or_size_splits
    assert isinstance(sizes, (tuple, list))
    outputs = []
    for i, size in enumerate(sizes):
      scope_ = scope + '_{}'.format(i + 1)
      outputs.append(self.dense_rn(
        x, s, scope_, activation=activation, output_dim=size, is_gate=is_gate,
        layer_normalization=True))
    return outputs



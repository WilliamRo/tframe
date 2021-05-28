from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe import checker
from tframe import linker
from tframe import hub as th
from tframe.nets.rnn_cells.cell_base import CellBase

from tframe.operators.apis.distributor import Distributor


class GDU(CellBase, Distributor):
  """Grouped Distributor Unit
     Reference: gdu2019"""
  net_name = 'gdu'

  def __init__(
      self,
      configs,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      bias_initializer='zeros',
      reverse=False,
      use_reset_gate=False,
      dropout=0.0,
      layer_normalization=False,
      **kwargs):
    """
    :param configs: a list or tuple of tuples with format (size, num, delta)
                    or a string with format `S1xM1xD1+S2xM2xD2+...`
    """
    # Call parent's constructor
    CellBase.__init__(self, activation, weight_initializer, use_bias,
                      bias_initializer, layer_normalization, **kwargs)

    # Specific attributes
    self._reverse = checker.check_type(reverse, bool)
    self._use_reset_gate = checker.check_type(use_reset_gate, bool)

    self._groups = self._get_groups(configs)
    self._state_size = self._get_total_size(self._groups)
    self._dropout_rate = checker.check_type(dropout, float)
    assert 0 <= dropout < 1
    # matrices for SOG v1
    self._D = None
    self._S = None


  @property
  def _scale_tail(self):
    config_str = self._get_config_string(self._groups, reverse=self._reverse)
    tail = '({})'.format(config_str)
    if self._use_reset_gate: tail = '[r]' + tail
    return tail


  @staticmethod
  def mark():
    return '{}gdu({})'.format('r' if th.use_reset_gate else '', th.gdu_string)


  def _get_sog_activation(self, x, s, configs, scope, name):
    assert isinstance(configs, (list, tuple)) and len(configs) > 0
    net_u = self.dense_rn(x, s, scope)
    if th.sog_version == 0: u = linker.softmax_over_groups(net_u, configs, name)
    else: u = self._sog_v1(net_u)
    return u


  def _link(self, prev_s, x, **kwargs):
    self._check_state(prev_s)

    # - Calculate update gates
    u, z = self._get_coupled_gates(
      x, prev_s, self._groups, reverse=self._reverse)
    self._gate_dict['beta_gate'] = z

    # - Calculate s_bar
    if self._use_reset_gate:
      s_bar, r = self.reset_14(
        x, prev_s, 's_bar', self._activation, reset_s=True, return_gate=True)
      self._gate_dict['reset_gate'] = r
    else:
      s_bar = self.dense_rn(x, prev_s, 's_bar', self._activation)

    # - Dropout if necessary
    if self._dropout_rate > 0: s_bar = self.dropout(s_bar, self._dropout_rate)

    # - Update state
    with tf.name_scope('transit'):
      new_s = z * prev_s + u * s_bar

    # - Calculate output and return
    y = new_s
    return y, new_s


  def _sog_v1(self, x):
    # This version of sog is much faster than v0
    s, n, delta = self._groups[0]
    assert len(self._groups) == 1 and delta == 1
    # Check matrices
    if self._D is None:
      # This code block is copied from GAM._init_const_matrices
      # Duplicating matrix D
      D = np.zeros((n, s * n), dtype=np.float32)
      indices=[[i, j] for i in range(n) for j in range(i*s, i*s+s)]
      for i, j in indices: D[i, j] = 1.0
      self._D = tf.constant(D, dtype=th.dtype)
      # Summarizing matrix S
      S = np.transpose(D)
      self._S = tf.constant(S, dtype=th.dtype)
    # Calculate SOG
    exp = tf.exp(x)
    deno = tf.matmul(tf.matmul(exp, self._S), self._D)
    return tf.divide(exp, deno, name='sog')

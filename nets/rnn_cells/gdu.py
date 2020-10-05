from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker
from tframe import linker
from tframe.nets.rnn_cells.cell_base import CellBase

from tframe.operators.apis.distributor import Distributor
from tframe.operators.apis.dynamic_weights import DynamicWeights


class GDU(CellBase, Distributor, DynamicWeights):
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
      reset_who='s',
      shunt_output=False,
      gate_output=False,
      **kwargs):
    """
    :param configs: a list or tuple of tuples with format (size, num, delta)
                    or a string with format `S1xM1xD1+S2xM2xD2+...`
    """
    # Call parent's constructor
    CellBase.__init__(self, activation, weight_initializer,
                      use_bias, bias_initializer, **kwargs)

    # Specific attributes
    self._reverse = checker.check_type(reverse, bool)
    self._use_reset_gate = checker.check_type(use_reset_gate, bool)
    self._shunt_output = checker.check_type(shunt_output, bool)
    self._gate_output = checker.check_type(gate_output, bool)

    self._groups = self._get_groups(configs)
    self._state_size = self._get_total_size(self._groups)

    assert reset_who in ('a', 's')
    self._reset_who = reset_who


  @property
  def _scale_tail(self):
    config_str = self._get_config_string(self._groups, reverse=self._reverse)
    if self._use_reset_gate: config_str += '|r' + self._reset_who
    tail = '({})'.format(config_str)
    if self._shunt_output: tail += '[{}]'.format(self._state_size)
    return tail


  def _get_sog_activation(self, x, s, configs, scope, name):
    assert isinstance(configs, (list, tuple)) and len(configs) > 0
    net_u = self.dense_rn(x, s, scope=scope)
    u = linker.softmax_over_groups(net_u, configs, name)
    return u


  def _link(self, prev_s, x, **kwargs):
    self._check_state(prev_s)

    # - Calculate update gates
    u, z = self._get_coupled_gates(
      x, prev_s, self._groups, reverse=self._reverse)
    self._gate_dict['beta_gate'] = z

    # - Calculate s_bar
    if self._use_reset_gate:
      s_bar = self.neurons_with_reset_gate(x, prev_s, self._reset_who)
    else: s_bar = self.dense_rn(
      x, prev_s, activation=self._activation, scope='s_bar')

    # - Update state
    with tf.name_scope('transit'):
      new_s = tf.add(tf.multiply(z, prev_s), tf.multiply(u, s_bar))

    # - Calculate output and return
    y = new_s
    if self._shunt_output:
      r = self.neurons(x, prev_s, is_gate=True, scope='y_reset_gate')
      self._gate_dict['y_reset_gate'] = r
      s = tf.multiply(r, prev_s)
      y = self.neurons(x, s, activation=self._activation, scope='output')
    elif self._gate_output:
      og = self.neurons(x, prev_s, is_gate=True, scope='output_gate')
      self._gate_dict['output_gate'] = og
      y = tf.multiply(og, y)

    return y, new_s


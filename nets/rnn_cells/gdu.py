from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker
from tframe import linker
from tframe.nets.rnn_cells.cell_base import CellBase

from tframe.utils.apis.distributor import Distributor


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

    self._groups = self._get_groups(configs)
    self._state_size = self._get_total_size(self._groups)


  @property
  def _scale_tail(self):
    return '({})'.format(
      self._get_config_string(self._groups, reverse=self._reverse))


  def _get_coupled_gates(self, x, s, configs, reverse):
    assert isinstance(configs, (list, tuple)) and len(configs) > 0
    # u for update, z for zone-out
    net_u = self.neurons(x, s, scope='net_u')
    u = linker.softmax_over_groups(net_u, configs, 'u_gate')
    z = tf.subtract(1., u, name='z_gate')
    if reverse: u, z = z, u
    self._gate_dict['beta_gate'] = z
    return u, z


  def _link(self, prev_s, x, **kwargs):
    self._check_state(prev_s)

    # - Calculate update gates
    u, z = self._get_coupled_gates(
      x, prev_s, self._groups, reverse=self._reverse)
    # - Calculate s_bar
    s = prev_s
    if self._use_reset_gate:
      r = self.neurons(x, s, is_gate=True, scope='reset_gate')
      self._gate_dict['reset_gate'] = r
      s = tf.multiply(r, s)
    s_bar = self.neurons(x, s, activation=self._activation, scope='s_bar')
    # - Update state
    with tf.name_scope('transit'):
      new_s = tf.add(tf.multiply(z, prev_s), tf.multiply(u, s_bar))

    # - Return new states
    y = new_s
    return y, new_s


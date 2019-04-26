from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import checker
from tframe import linker
from tframe.nets.rnn_cells.cell_base import CellBase


class GDU(CellBase):
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


  # region : Private Methods

  @staticmethod
  def _get_total_size(groups):
    assert isinstance(groups, (list, tuple))
    return int(sum([np.prod(g[0:2]) for g in groups]))

  @staticmethod
  def _get_config_string(groups, reverse):
    groups = list(groups)
    for i, g_ in enumerate(groups):
      g = list(g_)
      g[:2] = [str(n) for n in g[:2]]
      # if delta = 1, hide it; if delta = -1, replace it with `s`
      if g[-1] == 1: g.pop(-1)
      elif g[-1] == -1: g[-1] = 'S'
      else: g[-1] = '{:.2f}'.format(g[-1])
      # Set g back to groups
      groups[i] = g
    # Add reverse token if necessary
    s = '+'.join(['x'.join(g) for g in groups])
    if reverse: s += '|r'
    return s

  @staticmethod
  def _get_groups(configs):
    # Parse config string if necessary
    if isinstance(configs, str):
      configs = GDU._parse_config_string(configs)
    # Check configs
    assert isinstance(configs, (list, tuple))
    configs = list(configs)
    for i, c in enumerate(configs):
      assert isinstance(c, (tuple, list))
      c = list(c)
      if len(c) == 2: c.append(1.0)
      assert len(c) == 3
      checker.check_positive_integer(c[0])
      checker.check_positive_integer(c[1])
      assert isinstance(c[2], (int, float)) and (0 < c[2] <= c[0] or c[2] == -1)
      configs[i] = tuple(c)
    return tuple(configs)

  @staticmethod
  def _parse_config_string(config_string):
    assert isinstance(config_string, str)
    configs = []
    for s in config_string.split('+'):
      assert isinstance(s, str) and len(s) > 2
      c = [int(n) if i < 2 else float(n) for i, n in enumerate(s.split('x'))]
      configs.append(c)
    return configs

  # endregion : Private Methods




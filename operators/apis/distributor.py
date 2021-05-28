from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe import checker
from tframe.utils import linker


class Distributor(object):

  def __init__(self, configs):
    self._groups = self._get_groups(configs)

  @property
  def total_size(self):
    return self._get_total_size(self._groups)

  @property
  def total_groups(self):
    return sum([g[1] for g in self._groups])

  def _get_sog_activation(self, x, s, configs, scope, name):
    raise NotImplementedError

  def _get_coupled_gates(self, x, s, configs, reverse):
    assert isinstance(configs, (list, tuple)) and len(configs) > 0
    # u for update, z for zone-out
    u = self._get_sog_activation(
      x, s, configs, scope='net_u', name='z_gate' if reverse else 'u_gate')
    z = tf.subtract(1., u, name='u_gate' if reverse else 'z_gate')
    if reverse: u, z = z, u
    return u, z

  # region : Static Methods

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
      configs = Distributor._parse_config_string(configs)
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
    return tuple(sorted(configs, key=lambda g: g[0]))

  @staticmethod
  def _parse_config_string(config_string):
    assert isinstance(config_string, str)
    configs = []
    for s in config_string.split('+'):
      assert isinstance(s, str) and len(s) > 2
      c = [int(n) if i < 2 else float(n) for i, n in enumerate(s.split('x'))]
      configs.append(c)
    return configs

  @staticmethod
  def _softmax_over_groups(a, configs, output_name='sog'):
    return linker.softmax_over_groups(a, configs, output_name)

  def _distributively_write(self, net_a, data):
    assert linker.get_dimension(data) == self.total_groups
    splitted_net_a = linker.split(net_a, self._groups)
    splitted_data = linker.split_by_sizes(data, [g[1] for g in self._groups])

    a_list, bar_list = [], []
    for (s, n), net_a_, data_ in zip(
        [(g[0], g[1]) for g in self._groups], splitted_net_a, splitted_data):
      a = net_a
      bar = data_
      if n > 1:
        a = tf.reshape(a, [-1, s])
        bar = tf.reshape(bar, [-1, 1])
      a = tf.nn.softmax(a)
      bar = a * bar
      if n > 1:
        a = tf.reshape(a, [-1, s*n])
        bar = tf.reshape(bar, [-1, s*n])
      a_list.append(a)
      bar_list.append(bar)

    a = linker.concatenate(a_list)
    bar = linker.concatenate(bar_list)
    return a, bar

  # endregion : Static Methods

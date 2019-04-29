from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import checker


class Distributor(object):

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

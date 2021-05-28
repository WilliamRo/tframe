from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tframe import tf

from .neurobase import NeuroBase


class Mixer(NeuroBase):

  def __init__(self, config_string):

    self.groups = self._get_groups(config_string)


  @property
  def group_string(self):
    return '+'.join(['x'.join([str(n) for n in g]) for g in self.groups])

  @property
  def group_sizes(self):
    return [g[0] * g[1] for g in self.groups]

  @property
  def total_size(self):
    return sum(self.group_sizes)


  @staticmethod
  def _get_groups(config_string):
    assert isinstance(config_string, str)
    assert re.fullmatch(r'\d+x\d+(\+\d+x\d)*', config_string) is not None

    groups = []
    for gs in config_string.split('+'):
      assert isinstance(gs, str) and len(gs) > 2
      groups.append([int(n) for n in gs.split('x')])
    groups = tuple(sorted(groups, key=lambda g: g[0]))

    # Check groups
    assert len(set([g[0] for g in groups])) == len(groups)
    for g in groups: assert g[0] > 0 and g[1] > 0

    return groups


  def _cumax_over_groups(self, a):
    assert isinstance(a, tf.Tensor)

    group_sizes = [s*n + (n if s != 1 else 0) for s, n in self.groups]
    splitted = tf.split(a, group_sizes, axis=1) if len(group_sizes) > 1 else [a]

    output_list = []
    # s: group size; n: group number
    for (s, n), net_a in zip(self.groups, splitted):
      if s == 1:
        output_list.append(tf.sigmoid(net_a))
        continue
      if n > 1: net_a = tf.reshape(net_a, [-1, s+1])
      activated = tf.cumsum(tf.nn.softmax(net_a), axis=-1)[:, :-1]
      if n > 1: activated = tf.reshape(activated, [-1, s*n])
      output_list.append(activated)

    output = (tf.concat(output_list, axis=1, name='cog')
              if len(output_list) > 1 else output_list[0])
    return output





from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import checker
from tframe import linker

from .groups import Groups
from .neurobase import RNeuroBase


class HardDriver(Groups, RNeuroBase):
  """Prototype created on 2019-06-12"""

  def __init__(
      self,
      config_string,
      arm_size=None,
      fetching_method='sector',
  ):
    # Call parent's constructor
    Groups.__init__(self, config_string)

    # BETA attributes
    if arm_size is None: arm_size = self.num_groups
    self._arm_size = checker.check_positive_integer(arm_size)
    self._fetching_method = fetching_method


  def _read(self, arm, s, scope, num_heads=1):
    if self._fetching_method in ('sector', 'default'):
      return self._read_sector(arm, s, scope)
    elif self._fetching_method == 'mix':
      return self._read_mix(arm, s, scope)
    else: raise KeyError(
      '!! Unknown fetching method `{}`'.format(self._fetching_method))


  def _read_mix(self, arm, s, scope):
    # Prepare
    net_head = self.dense(
      self.total_size, arm, scope + '_read',
      bias_initializer='zeros', weight_initializer='glorot_normal')
    # Read
    def operator(state, net_h):
      return tf.reduce_sum(state * net_h, axis=1, keepdims=True)
    reshape2 = lambda _, n: n
    data = self._binary_operate_over_groups(
      s, net_head, operator, reshape2=reshape2)
    assert self.get_dimension(data) == self.num_groups
    return data


  def _read_sector(self, arm, s, scope):
    # Prepare
    net_head = self.dense(self.total_size, arm, scope + '_read')
    # Read
    head_list = []
    def operator(state, net_h, size):
      h = tf.nn.softmax(net_h, axis=1)
      head_list.append(tf.reshape(h, [-1, size]))
      return tf.reduce_sum(state * h, axis=1, keepdims=True)
    reshape2 = lambda _, n: n
    data = self._binary_operate_over_groups(
      s, net_head, operator, reshape2=reshape2)
    assert self.get_dimension(data) == self.num_groups
    # Concatenate head_list and register
    assert len(head_list) == len(self._groups)
    head = linker.concatenate(head_list)
    self._register_gate(scope + '_head', head)
    return data


  def _write(self, arm, s, s_bar):
    """Write rule:
        write_head: h
        data_to_write: data
        new_s = (1 - h) * s + prod_over_groups(h,  data)

    :param s_bar: tensor of shape [batch_size, num_groups]
    """

    # Get h and data
    net_head = self.dense(self.total_size, arm, 'net_head_write')
    new_data, head = self._distribute(s_bar, net_head)
    self._register_gate('write_head', head)
    return (1. - head) * s + new_data


  def _distribute(self, s_bar, net_h):
    """This method should be used only for hd-write methods"""
    head_list = []
    def operator(s_block, h_block, size):
      # s_block.shape = [batch_size*n, 1]
      #       h.shape = [batch_size*n, s]
      h = tf.nn.softmax(h_block)
      # y.shape = [batch_size*n, s]
      y = s_block * h
      head_list.append(tf.reshape(h, [-1, size]))
      return y
    reshape1_1 = lambda s, n: 1
    data = self._binary_operate_over_groups(
      s_bar, net_h, operator, sizes1=self.group_sizes, reshape1_1=reshape1_1)
    # Concatenate head_list to head
    assert len(head_list) == len(self._groups)
    head = linker.concatenate(head_list)
    return data, head


  def _register_gate(self, key, gate):
    gate_dict = getattr(self, '_gate_dict', None)
    if gate_dict is None: return
    gate_dict[key] = gate


  def _x_head(self, x, scope):
    net_head = self.dense(self.total_size, x, scope)
    return self._softmax_over_groups(net_head)


  def _x_heads(self, num_heads, x, scope):
    net_heads = self.dense(self.total_size * num_heads, x, scope)
    net_heads = tf.reshape(net_heads, shape=[-1, self.total_size])
    heads = self._softmax_over_groups(net_heads)
    heads = tf.reshape(heads, shape=[-1, self.total_size * num_heads])
    return tf.split(heads, num_heads, axis=1)


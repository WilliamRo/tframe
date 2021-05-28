from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe import checker
from tframe import hub
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
      diff_head=False,
      gutter=False,
      gutter_bias=None,
  ):
    # Call parent's constructor
    Groups.__init__(self, config_string)

    # BETA attributes
    if arm_size is None: arm_size = self.num_groups
    self._arm_size = checker.check_type(arm_size, int)
    self._fetching_method = fetching_method
    self._diff_head = checker.check_type(diff_head, bool)

    self._gutter = checker.check_type(gutter, bool)
    if gutter_bias is not None: assert isinstance(gutter_bias, (int, float))
    self._gutter_bias = gutter_bias


  def _read(self, arm, s, scope, num_heads=1):
    # Get read method
    if self._fetching_method in ('sector', 'default'):
      read_method = self._read_sector
    elif self._fetching_method == 'mix':
      read_method = self._read_mix
    else: raise KeyError(
      '!! Unknown fetching method `{}`'.format(self._fetching_method))
    # Read `num_head` times from hard-driver
    return read_method(arm, s, scope, num_heads=num_heads)


  def _read_mix(self, arm, s, scope, **kwargs):
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


  def _read_sector(self, arm, s, scope, num_heads):
    heads_sum = 0.
    data_list = []
    for i in range(num_heads):
      # Prepare
      net_head = self.dense(self.total_size, arm, scope + '_read_' + str(i))
      if self._diff_head:
        net_head, heads_sum = net_head - heads_sum, net_head + heads_sum
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
      data_list.append(data)
      # Concatenate head_list and register
      assert len(head_list) == len(self._groups)
      head = linker.concatenate(head_list)
      self._register_gate(scope + '_head_' + str(i), head)

    return linker.concatenate(data_list)


  def _write(self, arm, s, s_bar, tail='', gutter=False, return_head=False,
             full_write=False):
    """Write rule:
        write_head: h
        data_to_write: data
        new_s = (1 - h) * s + prod_over_groups(h,  data)

    :param s_bar: tensor of shape [batch_size, num_groups]
    """

    # - Get h and data
    head_size, b_init = self.total_size, None
    if gutter:
      head_size += self.num_groups
      if self._gutter_bias is not None: b_init = self._get_gutter_bias_init()
    net_head = self.dense(
      head_size, arm, 'net_head_write' + tail, bias_initializer=b_init)

    new_data, head = self._distribute(
      s_bar, net_head, gutter, full_write=full_write)
    self._register_gate('write_head' + tail, head)
    output = (1. - head) * s + new_data
    if return_head: return output, head
    return output


  def _write_v2(self, arm, s, s_bar, tail=''):
    net_head = self.dense(self.total_size, arm, 'net_head_write' + tail)


  def _distribute(self, s_bar, net_h, gutter=False, full_write=False):
    """This method should be used only for hd-write methods"""
    head_list = []
    def operator(s_block, h_block, size):
      # s_block.shape = [batch_size*n, 1]
      #       h.shape = [batch_size*n, s] or [batch_size*n, s+1]
      h = tf.nn.softmax(h_block)
      if gutter: h = h[:, :-1]
      # y.shape = [batch_size*n, s]
      y = s_block * h
      head_list.append(tf.reshape(h, [-1, size]))
      return y
    sizes1 = self.group_sizes if full_write else self.group_duplicates
    sizes2 = [s + 1 for s in self.group_sizes] if gutter else self.group_sizes
    reshape1_1 = lambda s, n: s if full_write else 1
    reshape1_2 = lambda s, n: s + int(gutter)
    data = self._binary_operate_over_groups(
      s_bar, net_h, operator, sizes1=sizes1, sizes2=sizes2,
      reshape1_1=reshape1_1, reshape1_2=reshape1_2)
    # Concatenate head_list to head
    assert len(head_list) == len(self._groups)
    head = linker.concatenate(head_list)
    return data, head


  def _get_gutter_bias_init(self):
    assert self._gutter
    if self._gutter_bias is None: return None
    b_list = []
    for s, n in self._groups:
      b = np.zeros(shape=[s*n+n], dtype=np.float32)
      for i in range(n):
        # For each group, set the last number to `bias`
        b[(i+1)*(s+1)-1] = self._gutter_bias
      b_list.append(b)
    bias = np.concatenate(b_list)
    return tf.initializers.constant(bias, dtype=hub.dtype)


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


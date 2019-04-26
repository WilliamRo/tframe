from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import activations
from tframe import checker
from tframe import context
from tframe import hub
from tframe import linker
from tframe import initializers
from tframe import regularizers

from tframe.nets import RNet


class Noah(RNet):
  """Vanilla RNN cell with a linear auxiliary memory.
  """
  net_name = 'noah'

  def __init__(
      self,
      state_size,
      mem_fc=True,
      **kwargs):
    # Call parent's constructor
    RNet.__init__(self, self.net_name)

    # Attributes
    self._state_size = state_size
    self._activation = activations.get('tanh', **kwargs)
    # self._use_bias = True
    self._weight_initializer = initializers.get('xavier_normal')
    self._bias_initializer = initializers.get('zeros')
    self._output_scale = state_size
    self._fully_connect_memories = mem_fc


  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    self._init_state = (self._get_placeholder('h', self._state_size),
                        self._get_placeholder('s', self._state_size))
    return self._init_state


  def structure_string(self, detail=True, scale=True):
    return self.net_name + '({})'.format(self._state_size) if scale else ''


  def _link(self, pre_states, x, **kwargs):
    h, s = pre_states
    x_size = self._get_size(x)

    # Calculate net_{xh}
    Wxh = self._get_variable(
      'Wxh', [self._state_size + x_size, self._state_size])
    x_h = tf.concat([x, h], axis=1)
    net_xh = tf.matmul(x_h, Wxh)

    # Calculate net_s
    if self._fully_connect_memories:
      Ws = self._get_variable('Ws', [self._state_size, self._state_size])
      net_s = tf.matmul(s, Ws)
    else:
      Ws = self._get_variable('Ws', self._state_size)
      net_s = tf.multiply(s, Ws)

    # Calculate new_h and new_s
    bias = self._get_variable('bias', self._state_size)
    net = tf.nn.bias_add(tf.add(net_xh, net_s), bias)
    new_h = self._activation(net, name='y')
    new_s = tf.add(s, new_h)

    return new_h, (new_h, new_s)


class Shem(RNet):
  net_name = 'shem'

  def __init__(
      self,
      state_size,
      activation='tanh',
      weight_initializer='xavier_normal',
      input_gate=True,
      forget_gate=True,
      output_gate=True,
      use_g_bias=True,
      g_bias_initializer='zeros',
      use_i_bias=True,
      i_bias_initializer='zeros',
      use_f_bias=True,
      f_bias_initializer='zeros',
      use_o_bias=True,
      o_bias_initializer='zeros',
      output_as_mem=True,
      fully_connect_memory=True,
      activate_memory=True,
      truncate_grad=False,
      **kwargs):
    # Call parent's constructor
    RNet.__init__(self, self.net_name)

    # Attributes
    self._state_size = state_size
    self._input_gate = checker.check_type(input_gate, bool)
    self._forget_gate = checker.check_type(forget_gate, bool)
    self._output_gate = checker.check_type(output_gate, bool)
    self._activation = activations.get(activation, **kwargs)
    self._weight_initializer = initializers.get(weight_initializer)
    self._use_g_bias = checker.check_type(use_g_bias, bool)
    self._g_bias_initializer = initializers.get(g_bias_initializer)
    self._use_i_bias = checker.check_type(use_i_bias, bool)
    self._i_bias_initializer = initializers.get(i_bias_initializer)
    self._use_f_bias = checker.check_type(use_f_bias, bool)
    self._f_bias_initializer = initializers.get(f_bias_initializer)
    self._use_o_bias = checker.check_type(use_o_bias, bool)
    self._o_bias_initializer = initializers.get(o_bias_initializer)
    self._activate_mem = checker.check_type(activate_memory, bool)
    self._truncate_grad = checker.check_type(truncate_grad, bool)
    self._fc_memory = checker.check_type(fully_connect_memory, bool)
    self._output_as_mem = checker.check_type(output_as_mem, bool)
    self._kwargs = kwargs


  def structure_string(self, detail=True, scale=True):
    return self.net_name + '({})'.format(self._state_size) if scale else ''


  def _link(self, s, x, **kwargs):

    def neurons(name, activation, use_bias, bias_init, mem=s):
      return self.neurons(
        x, mem, self._state_size, self._fc_memory, activation=activation,
        scope=name, truncate=self._truncate_grad, use_bias=use_bias,
        bias_initializer=bias_init)
    def gate(input_, name, use_bias, bias_init):
      gate_ = neurons(name, 'sigmoid', use_bias, bias_init)
      return tf.multiply(gate_, input_)

    # - Calculate memory
    # ..Calculate g
    g = neurons('g', self._activation, self._use_g_bias,
                self._g_bias_initializer)
    if self._input_gate: g = gate(
      g, 'input_gate', self._use_i_bias, self._i_bias_initializer)
    # ..Maybe forget
    s_prev = s
    if self._forget_gate: s_prev = gate(
      s, 'forget_gate', self._use_f_bias, self._f_bias_initializer)
    new_s = tf.add(s_prev, g)

    # - Calculate output
    s_out = self._activation(s) if self._activate_mem else s
    if self._output_gate: s_out = gate(
      s_out, 'output_gate', self._use_o_bias, self._o_bias_initializer)
    y = neurons('y', self._activation, True, 'zeros', s_out)

    return y, new_s


class Ham(RNet):
  """Based on Shem, Ham has multiple memory units."""
  net_name = 'Ham'

  class MemoryUnit(object):
    def __init__(self, size, in_gate, forget_gate, out_gate, act_mem, fc_mem):
      self.size = size

      self.use_input_gate = in_gate
      self.use_forget_gate = forget_gate
      self.use_output_gate = out_gate

      self.activate_memory = act_mem
      self.fully_connect_memory = fc_mem

    # region : Properties

    @property
    def detail_string(self):
      ds = '[{}-'.format(self.size)
      if self.use_input_gate: ds += 'i'
      if self.use_forget_gate: ds += 'f'
      if self.use_output_gate: ds += 'o'
      if str.isalnum(ds[-1]): ds += '-'
      if not self.activate_memory: ds += 'n'
      ds += 'a-'
      if not self.fully_connect_memory: ds += 'n'
      return ds + 'fc]'

    # endregion : Properties

    # region : Static Methods

    @staticmethod
    def parse_unit(config):
      separator = '-'
      assert isinstance(config, str)
      cfgs = config.split(separator)
      assert len(cfgs) >= 3
      # 1. Size
      size = int(cfgs[0])
      if size == 0: return None
      # 2. Activate memory before use
      act_mem = False
      if 'a' in cfgs: act_mem = True
      else: assert 'na' in cfgs
      # 3. Fully connect memory
      fc_mem = False
      if 'fc' in cfgs: fc_mem = True
      else: assert 'nfc' in cfgs
      # 4 - 6: input gate, forget gate, output gate
      in_gate = 'i' in cfgs
      forget_gate = 'f' in cfgs
      out_gate = 'o' in cfgs
      # Wrap and return
      return Ham.MemoryUnit(
        size, in_gate, forget_gate, out_gate, act_mem, fc_mem)

    @staticmethod
    def parse_units(config):
      separator = ';'
      assert isinstance(config, str)
      cfgs = config.split(separator)
      assert len(cfgs) >= 1
      units = []
      for cfg in cfgs:
        unit = Ham.MemoryUnit.parse_unit(cfg)
        if unit is not None: units.append(unit)
      return units

    # endregion : Static Methods

  def __init__(self, output_dim,
               memory_units=None, mem_config=None,
               use_mem_wisely=False,
               weight_regularizer=None,
               **kwargs):
    # Call parent's constructor
    RNet.__init__(self, self.net_name)

    # Attributes
    self.output_dim = output_dim
    self.memory_units = (self.MemoryUnit.parse_units(mem_config) if
                         memory_units is None else memory_units)
    self.memory_units = [mu for mu in self.memory_units if mu.size > 0]
    checker.check_type(self.memory_units, Ham.MemoryUnit)

    self._state_size = sum([mu.size for mu in self.memory_units])
    self._activation = activations.get('tanh', **kwargs)

    self._use_mem_wisely = use_mem_wisely
    self._truncate = kwargs.get('truncate', False)
    self._weight_regularizer = regularizers.get(weight_regularizer, **kwargs)
    # self._use_global_reg = kwargs.get('global_reg', False)
    self._kwargs = kwargs

  @staticmethod
  def recommended_cell_for_erg(output_size, short_size=8, long_size=4):
    # Initialize memory unites
    short_memory_unit = Ham.MemoryUnit(
      size=short_size,
      in_gate=True,
      forget_gate=True,
      out_gate=False,
      act_mem=True,
      fc_mem=True)
    long_memory_unit = Ham.MemoryUnit(
      size=long_size,
      in_gate=True,
      forget_gate=False,
      out_gate=False,
      act_mem=True,
      fc_mem=False)
    cell = Ham(output_dim=output_size,
               memory_units=[short_memory_unit, long_memory_unit],
               use_mem_wisely=True)
    return cell

  @property
  def detail_string(self):
    return ''.join([m.detail_string for m in self.memory_units])

  @property
  def gate_number(self):
    return sum([mu.use_input_gate + mu.use_forget_gate + mu.use_output_gate
                for mu in self.memory_units])

  def structure_string(self, detail=True, scale=True):
    if detail: scale_string = '_{}'.format(self.output_dim)
    else: scale_string = '[{}]_{}'.format(
      '+'.join([str(m.size) for m in self.memory_units]), self.output_dim)

    if scale: return self.net_name + scale_string
    else: return self.net_name

  # region : Private Methods

  def _link(self, pre_s_block, x, **kwargs):
    def forward(name, memory, fc_mem, output_dim, activation=self._activation):
      assert output_dim is not None
      return self._neurons_forward_with_memory(
        x, memory, name, activation, fc_mem, output_dim, use_bias=True,
        truncate=self._truncate, w_reg=self._weight_regularizer)

    def gate(name, tensor, memory, fc_mem, gate_name=None):
      multiply = linker.get_multiply(self._truncate)
      g = forward(
        name, memory, fc_mem, output_dim=self._get_size(tensor),
        activation=tf.sigmoid)
      if gate_name is not None:
        self._gate_dict[gate_name] = g
        # context.add_tensor_to_export(gate_name, g)
        # context.add_to_dict_collection(self.GATES_ACTIVATIONS, gate_name, g)
      return multiply(g, tensor)

    # Prepare splitted memory
    pre_s_list = self._split_memory(pre_s_block)

    # Calculate new_s and s_out
    assert isinstance(self.memory_units, list)
    new_s_list, s_out_list = [], []
    for i, mu, s in zip(range(len(pre_s_list)), self.memory_units, pre_s_list):
      if mu.use_input_gate or mu.use_forget_gate or mu.use_output_gate:
        if self._use_mem_wisely:
          if i == 0 or i == 1: mem = pre_s_list[0]
          else: mem = tf.concat(pre_s_list[:i], axis=1)
        else: mem = s
      fc_mem = mu.fully_connect_memory
      # New memory
      with tf.variable_scope('memory_unit_{}'.format(i + 1)):
        s_bar = forward('s_bar', s, fc_mem, mu.size)
        # Input gate
        gate_name = 'in_gate_{}'.format(i + 1) if hub.export_gates else None
        if mu.use_input_gate: s_bar = gate(
          'in_gate', s_bar, mem, fc_mem or self._use_mem_wisely, gate_name)
        # Forget gate
        gate_name = 'forget_gate_{}'.format(i + 1) if hub.export_gates else None
        prev_s = (
          gate('forget_gate', s, mem, fc_mem or self._use_mem_wisely, gate_name)
          if mu.use_forget_gate else s)
        # Add prev_s and s_bar
        new_s = tf.add(prev_s, s_bar)
      new_s_list.append(new_s)

      # Memory for calculating output y
      with tf.variable_scope('s_out_{}'.format(i + 1)):
        s_out = self._activation(s) if mu.activate_memory else s
        gate_name = 'out_gate_{}'.format(i + 1) if hub.export_gates else None
        if mu.use_output_gate: s_out = gate(
          'out_gate', s_out, mem, fc_mem or self._use_mem_wisely, gate_name)
      s_out_list.append(s_out)

    new_s = tf.concat(new_s_list, axis=1, name='new_s')
    s_out = tf.concat(s_out_list, axis=1, name='s_out')

    # Calculate output
    y = forward('output', s_out, True, self.output_dim, self._activation)

    return y, new_s

  def _split_memory(self, s):
    assert isinstance(s, tf.Tensor)
    size_splits = [m.size for m in self.memory_units]
    return tf.split(s, num_or_size_splits=size_splits, axis=1)

  # endregion : Private Methods

  # region : Public Methods

  # endregion : Public Methods


class Japheth(RNet):
  net_name = 'Japheth'


  def structure_string(self, detail=True, scale=True):
    return self.net_name + '({})'.format(self._state_size) if scale else ''

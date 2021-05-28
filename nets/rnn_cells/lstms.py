from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe import activations
from tframe import initializers
from tframe import hub

from tframe.nets import RNet
from tframe.nets.rnn_cells.cell_base import CellBase


class LSTM(CellBase):
  """The most common variant of LSTMs"""

  net_name = 'lstm'

  def __init__(
      self,
      state_size,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      couple_fi=False,
      cell_bias_initializer='zeros',
      input_bias_initializer='zeros',
      output_bias_initializer='zeros',
      forget_bias_initializer='zeros',
      use_output_activation=True,
      **kwargs):
    # Call parent's constructor
    CellBase.__init__(self, activation, weight_initializer,
                      use_bias, cell_bias_initializer, **kwargs)

    # Specific attributes
    self._state_size = checker.check_positive_integer(state_size)
    self._input_bias_initializer = initializers.get(input_bias_initializer)
    self._output_bias_initializer = initializers.get(output_bias_initializer)
    self._forget_bias_initializer = initializers.get(forget_bias_initializer)

    self._couple_fi = checker.check_type(couple_fi, bool)
    self._use_output_activation = checker.check_type(
      use_output_activation, bool)


  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    get_placeholder = lambda name: self._get_placeholder(name, self._state_size)
    self._init_state = (get_placeholder('h'), get_placeholder('c'))
    return self._init_state


  def _link(self, pre_states, x, **kwargs):
    self._check_state(pre_states, 2)
    h, c = pre_states

    # Get f, i, o, g
    if self.lottery_activated:f, i, o, g = self._get_fiog(x, h)
    else: f, i, o, g = self._get_fiog_fast(x, h)

    # - Update state
    # Apply recurrent dropout if necessary
    if self._dropout_rate > 0: g = self.dropout(g, self._dropout_rate)
    # Calculate new_c
    new_c = tf.add(tf.multiply(f, c), tf.multiply(i, g), 'new_c')
    # Calculate new_h
    new_c_tilde = tf.tanh(new_c) if self._use_output_activation else new_c
    new_h = tf.multiply(o, new_c_tilde)

    # Register gates and return
    self._gate_dict['input_gate'] = i
    self._gate_dict['forget_gate'] = f
    self._gate_dict['output_gate'] = o
    return new_h, (new_h, new_c)


  def _get_fiog(self, x, h):
    f = self.neurons(x, h, is_gate=True, scope='f',
                     bias_initializer=self._forget_bias_initializer)
    if self._couple_fi: i = tf.subtract(1.0, f)
    else: i = self.neurons(x, h, is_gate=True, scope='i',
                           bias_initializer=self._input_bias_initializer)
    o = self.neurons(x, h, is_gate=True, scope='o',
                     bias_initializer=self._output_bias_initializer)
    g = self.neurons(x, h, scope='g', activation=self._activation,
                     bias_initializer=self._bias_initializer)
    return f, i, o, g


  def _get_fiog_fast(self, x, h):
    # Calculate net inputs
    if self._couple_fi: net_f, net_o, net_g = self.neurons(
        x, h, scope='net_inputs', num_or_size_splits=3, use_bias=False)
    else: net_f, net_i, net_o, net_g = self.neurons(
        x, h, scope='net_inputs', num_or_size_splits=4, use_bias=False)

    # Get f, i, o, g
    f = self._activate(net_f, self._forget_bias_initializer, 'f')
    if self._couple_fi: i = tf.subtract(1.0, f)
    else: i = self._activate(net_i, self._input_bias_initializer, 'i')
    o = self._activate(net_o, self._output_bias_initializer, 'o')
    g = self._activate(net_g, self._bias_initializer, 'g')
    return f, i, o, g


  def _activate(self, net_input, bias_initializer, name):
    output = self.add_bias(net_input, bias_initializer, name)
    if name in ('i', 'f', 'o'): return tf.sigmoid(output)
    else: assert name == 'g'
    return self._activation(output)


class BasicLSTMCell(RNet):
  """Basic LSTM cell
  """
  net_name = 'basic_lstm'

  def __init__(
      self,
      state_size,
      activation='tanh',
      use_bias=True,
      weight_initializer='xavier_normal',
      bias_initializer='zeros',
      input_gate=True,
      output_gate=True,
      forget_gate=True,
      input_bias_initializer='zeros',
      output_bias_initializer='zeros',
      forget_bias_initializer='zeros',
      with_peepholes=False,
      truncate_grad=False,
      **kwargs):
    """
    :param state_size: state size: positive int
    :param activation: activation: string or callable
    :param use_bias: whether to use bias
    :param weight_initializer: weight initializer identifier
    :param bias_initializer: bias initializer identifier
    """
    # Call parent's constructor
    RNet.__init__(self, BasicLSTMCell.net_name)

    # Attributes
    self._state_size = state_size
    self._activation = activations.get(activation, **kwargs)
    self._use_bias = checker.check_type(use_bias, bool)
    self._weight_initializer = initializers.get(weight_initializer)
    self._bias_initializer = initializers.get(bias_initializer)
    self._input_bias_initializer = initializers.get(input_bias_initializer)
    self._output_bias_initializer = initializers.get(output_bias_initializer)
    self._forget_bias_initializer = initializers.get(forget_bias_initializer)

    self._input_gate = checker.check_type(input_gate, bool)
    self._output_gate = checker.check_type(output_gate, bool)
    self._forget_gate = checker.check_type(forget_gate, bool)
    self._with_peepholes = checker.check_type(with_peepholes, bool)
    self._truncate_grad = checker.check_type(truncate_grad, bool)

    self._output_scale = state_size

  # region : Properties

  def structure_string(self, detail=True, scale=True):
    gates = '[{}{}{}g{}]'.format('i' if self._input_gate else '',
                                 'f' if self._forget_gate else '',
                                 'o' if self._output_gate else '',
                                 'p' if self._with_peepholes else '')
    return self.net_name + gates + '({})'.format(
      self._state_size) if scale else ''

  @property
  def init_state(self):
    """Concatenate H and C together for the convenience to calculate dL/dS
    """
    if self._init_state is not None: return self._init_state

    assert self._state_size is not None
    self._init_state = self._get_placeholder('h_c', 2 * self._state_size)
    return self._init_state
    # get_placeholder = lambda name: self._get_placeholder(name, self._state_size)
    # self._init_state = (get_placeholder('h'), get_placeholder('c'))
    # return self._init_state

  # endregion : Properties

  # region : Private Methods

  def _link(self, pre_states, input_, **kwargs):
    # """pre_states = (h_{t-1}, c_{t-1})"""
    """pre_states = concat(h_{t-1}, c_{t-1})"""
    # self._check_state(pre_states, 2)
    self._check_state(pre_states, (2 * self._state_size,))
    # h, c = pre_states
    h, c = tf.split(pre_states, 2, axis=1)

    # Link
    if self._with_peepholes:
      new_h, new_c, (i, f, o) = self._link_with_peepholes(input_, h, c)
    else: new_h, new_c, (i, f, o) = self._basic_link(input_, h, c)

    # Register gates
    for k, g in zip(('input_gate', 'forget_gate', 'output_gate'), (i, f, o)):
      if g is not None: self._gate_dict[k] = g

    # Return a tuple with the same structure as input pre_states
    # return new_h, (new_h, new_c)
    return new_h, tf.concat([new_h, new_c], axis=1, name='new_h_c')

  # endregion : Private Methods

  # region : Link Cores

  def _basic_link(self, x, h, c):
    # Determine size_splits according to gates to be used
    size_splits = 1 + self._input_gate + self._output_gate + self._forget_gate
    def _add_bias(tensor, initializer, name):
      if self._use_bias: return self.add_bias(tensor, initializer, name)
      else: return tensor

    # i = input_gate, g = new_input, f = forget_gate, o = output_gate
    i, f, o = (None,) * 3
    splits = list(self.neurons(
      x, h, scope='net_input_chunk', num_or_size_splits=size_splits,
      truncate=self._truncate_grad, use_bias=False))
    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    # - Calculate candidates to write
    g = self._activation(_add_bias(splits.pop(0), self._bias_initializer, 'g'))

    if self._input_gate:
      with tf.name_scope('write_gate'):
        i = tf.sigmoid(
          _add_bias(splits.pop(0), self._input_bias_initializer, 'i'))
        g = tf.multiply(i, g)
    # - Forget
    if self._forget_gate:
      with tf.name_scope('forget_gate'):
        f = tf.sigmoid(
          _add_bias(splits.pop(0), self._forget_bias_initializer, 'f'))
        c = tf.multiply(f, c)
    # - Write
    with tf.name_scope('write'): new_c = tf.add(c, g)
    # - Read
    new_h = self._activation(new_c)
    if self._output_gate:
      with tf.name_scope('read_gate'):
        o = tf.sigmoid(
          _add_bias(splits.pop(0), self._output_bias_initializer, 'o'))
        new_h = tf.multiply(o, new_h)
    assert len(splits) == 0

    return new_h, new_c, (i, f, o)

  def _link_with_peepholes(self, x, h, c):
    i, f, o = (None,) * 3
    # - Calculate g
    g = self.neurons(x, h, activation=self._activation, scope='g',
                     truncate=self._truncate_grad)
    # :: Calculate gates
    num_splits = self._forget_gate +  self._input_gate
    if num_splits > 0:
      x_h_c = tf.concat([x, h, c], axis=1)
      splits = list(self.neurons(
        x_h_c, scope='fi_chunk', is_gate=True, truncate=self._truncate_grad,
        num_or_size_splits=num_splits))
      # - Forget
      if self._forget_gate:
        with tf.name_scope('forget_gate'):
          f = splits.pop(0)
          c = tf.multiply(f, c)
      # - Input gate
      if self._input_gate:
        with tf.name_scope('input_gate'):
          i = splits.pop(0)
          g = tf.multiply(i, g)
    # - Write
    with tf.name_scope('write'):
      new_c = tf.add(c, g)
    # - Read
    new_h = self._activation(new_c)
    if self._output_gate:
      o = self.neurons(tf.concat([x, h, new_c], axis=1), is_gate=True,
                       scope='o', truncate=self._truncate_grad)
      new_h = tf.multiply(o, new_h)

    return new_h, new_c, (i, f, o)

  # endregion : Link Cores


class OriginalLSTMCell(RNet):
  """The original LSTM cell proposed in 1997, which has no forget gate and
     and cell blocks are of size 1
  """
  net_name = 'origin_lstm'

  def __init__(
      self,
      state_size,
      cell_activation='sigmoid',                # g
      cell_activation_range=(-2, 2),
      memory_activation='sigmoid',              # h
      memory_activation_range=(-1, 1),
      weight_initializer='random_uniform',
      weight_initial_range=(-0.1, 0.1),
      use_cell_bias=False,
      cell_bias_initializer='random_uniform',
      cell_bias_init_range=(-0.1, 0.1),
      use_in_bias=True,
      in_bias_initializer='zeros',
      use_out_bias=True,
      out_bias_initializer='zeros',
      truncate=True,
      forward_gate=True,
      **kwargs):

    # Call parent's constructor
    RNet.__init__(self, OriginalLSTMCell.net_name)

    # Set state size
    self._state_size = state_size

    # Set activation
    # .. In LSTM98, cell activation is referred to as 'g',
    # .. while memory activation is 'h' and gate activation is 'f'
    self._cell_activation = activations.get(
      cell_activation, range=cell_activation_range)
    self._memory_activation = activations.get(
      memory_activation, range=memory_activation_range)
    self._gate_activation = activations.get('sigmoid')

    # Set weight and bias configs
    self._weight_initializer = initializers.get(
      weight_initializer, range=weight_initial_range)
    self._use_cell_bias = use_cell_bias
    self._cell_bias_initializer = initializers.get(
      cell_bias_initializer, range=cell_bias_init_range)
    self._use_in_bias = use_in_bias
    self._in_bias_initializer = initializers.get(in_bias_initializer)
    self._use_out_bias = use_out_bias
    self._out_bias_initializer = initializers.get(out_bias_initializer)

    if kwargs.get('rule97', False):
      self._cell_bias_initializer = self._weight_initializer
      self._in_bias_initializer = self._weight_initializer

    # Additional options
    self._truncate = truncate
    self._forward_gate = forward_gate

    # ...
    self._num_splits = 3
    self._output_scale = state_size
    self._h_size = (state_size * self._num_splits if self._forward_gate else
                    state_size)

    # TODO: BETA
    self.compute_gradients = self.truncated_rtrl

  # region : Properties

  def structure_string(self, detail=True, scale=True):
    str = self.net_name
    if scale: str += '({})'.format(self._state_size)
    return str

  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    assert self._state_size is not None
    self._init_state = (self._get_placeholder('h', self._h_size),
                        self._get_placeholder('c', self._state_size))
    return self._init_state

  @property
  def gradient_buffer_placeholder(self):
    if self._gradient_buffer_placeholder is not None:
      return self._gradient_buffer_placeholder
    # This property must be called after model is linked so that self.parameters
    #   can be accessed
    assert self.linked
    buffer = []
    for W in self.custom_var_list:
      shape = [None] + W.shape.as_list() + [self._state_size]
      buffer.append(tf.placeholder(dtype=hub.dtype, shape=shape))

    self._gradient_buffer_placeholder = tuple(buffer)
    return self._gradient_buffer_placeholder

  # endregion : Properties

  # region : Private Methods

  def _link(self, u_and_s, x, **kwargs):
    # Unpack previous states and prepare activation functions
    self._check_state(u_and_s, (self._h_size, self._state_size))
    u, s = u_and_s
    f = self._gate_activation
    h = self._memory_activation
    g = self._cell_activation
    def neurons(activation, scope, use_bias, bias_initializer):
      return self.neurons(
        x, u, num=self._state_size, activation=activation, scope=scope,
        truncate=self._truncate, weight_initializer=self._weight_initializer,
        use_bias=use_bias, bias_initializer=bias_initializer)

    # Calculate y^{in}(t)
    y_in = neurons(
      f, 'in_gate', self._use_in_bias, self._in_bias_initializer)
    # Calculate y^{out}(t)
    y_out = neurons(
      f, 'out_gate', self._use_out_bias, self._out_bias_initializer)
    # Calculate g(net_c(t))
    g_net_c = neurons(
      g, 'g_net_c', self._use_cell_bias, self._cell_bias_initializer)
    # Calculate the internal state s and y_c
    new_s = tf.add(s, tf.multiply(y_in, g_net_c))
    y_c = tf.multiply(y_out, h(new_s))

    # Wrap gate units into u if necessary
    new_u = tf.concat([y_c, y_in, y_out], axis=1) if self._forward_gate else y_c

    # Register gates
    for key, gate in zip(('input_gate', 'output_gate'), (y_in, y_out)):
      if gate is not None: self._gate_dict[key] = gate

    # TODO: BETA
    # self.repeater_tensors = new_c
    # self._custom_vars = [Wci] + [b for b in (in_bias, cell_bias)
    #                              if b is not None]
    return y_c, (new_u, new_s)

  # endregion : Private Methods

  # region : Public Methods

  def truncated_rtrl(self, dL_dy, **kwargs):
    """The origin LSTM_97 real-time recurrent training algorithm
    :param dL_dy: dL/dy with shape (num_steps(=1), batch_size, *y.shape)
    :return: (grads_and_vars, dS/dW)
    """
    # Step 0: Split new_c outside the scan loop
    S = self._new_state_tensor[1]
    state_size = S.shape[1]
    Ss = tf.split(S, num_or_size_splits=state_size, axis=1)

    # Step 1: Update dS/dW = (dS/dW1, ..., dS/dWm)
    mascot = tf.placeholder(tf.float32)
    dS_dW = []
    for dS_dWj_tau, Wj in zip(self.gradient_buffer_placeholder,
                              self.custom_var_list):
      dS_dWj = []
      split_dS_dWj_tau = tf.split(
        dS_dWj_tau, num_or_size_splits=state_size, axis=-1)
      for dSi_dWj_tau, Si in zip(split_dS_dWj_tau, Ss):
        dSi_dWj = []
        for b in range(hub.batch_size):
          grad = tf.gradients(Si[b], Wj)[0]
          dSi_dWj_n = dSi_dWj_tau[b] + tf.expand_dims(grad, -1)
          dSi_dWj.append(tf.expand_dims(dSi_dWj_n, 0))
        # Concatenate along batches
        dSi_dWj = tf.concat(dSi_dWj, axis=0) # (B, *W, 1)
        dS_dWj.append(dSi_dWj)
      dS_dWj = tf.concat(dS_dWj, axis=-1) # (B, *W, S)
      dS_dW.append(dS_dWj)
    dS_dW = tuple(dS_dW)

    # Step 2: Compute dL/dW as dL/dy * dy/dS * dS/dW
    #           = \sum_{n over batches} \sum_{k over states} ...
    #               dL/dy * dy/dS * dS/dW
    #    (1) dL_dy.shape = (?(=1), B, D) in dL_dy
    #    (2) dy_dS = (dy1/dS, ..., dyn/dS) in self._grad_tensors
    #TODO
    dL_dy = tf.reshape(dL_dy, shape=(-1, 1, dL_dy.shape[2]))  # (B, 1, D)
    dy_dS = self._grad_tensors # (B, D, S)
    def calc_dL_dW(_, mass):
      dldy, dyds, dsdw = mass
      # Calculate dL/dS
      dlds = tf.matmul(dldy, dyds)  # (1, S)
      # Calculate dL/dW
      dL_dW = []
      for dsdwj in dsdw:  # (*wj.shape, S)
        dL_dW.append(tf.reduce_sum(tf.multiply(dsdwj, dlds), axis=-1))
      return tuple(dL_dW)
    dL_dS_batch = tf.scan(calc_dL_dW, (dL_dy, dy_dS, dS_dW),
                          initializer=(mascot,) * len(self.custom_var_list))
    dL_dS = [tf.reduce_sum(t, axis=0) for t in dL_dS_batch]

    # Step 3: Return (((dW1, W1), ..., (dWn, Wn)), dS/dW)
    grads_and_vars = [(g, v) for g, v in zip(dL_dS, self.custom_var_list)]
    return tuple(grads_and_vars), dS_dW

  # endregion : Public Methods









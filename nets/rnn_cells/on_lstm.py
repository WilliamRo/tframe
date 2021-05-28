from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe import activations

from tframe.nets.rnn_cells.cell_base import CellBase


class ON_LSTM(CellBase):
  """ordered neurons LSTM, 2019"""

  net_name = 'on_lstm'

  def __init__(
      self,
      state_size,
      activation='tanh',
      weight_initializer='xavier_normal',
      use_bias=True,
      cell_bias_initializer='zeros',
      **kwargs):

    # Call parent's constructor
    CellBase.__init__(self, activation, weight_initializer,
                      use_bias, cell_bias_initializer, **kwargs)

    # Specific attributes
    self._state_size = checker.check_positive_integer(state_size)


  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    get_placeholder = lambda name: self._get_placeholder(name, self._state_size)
    self._init_state = (get_placeholder('h'), get_placeholder('c'))
    return self._init_state


  def _link(self, pre_states, x, **kwargs):
    self._check_state(pre_states, 2)
    h, c = pre_states

    # Get neuron activations
    if self.lottery_activated:
      f_tilde, i_tilde_bar, f, i, o, c_hat = self._get_neurons(x, h)
    else: f_tilde, i_tilde_bar, f, i, o, c_hat = self._get_neurons_fast(x, h)
    f_tilde_bar = tf.subtract(1., f_tilde)
    i_tilde = tf.subtract(1., i_tilde_bar)

    # Update
    # omega = tf.multiply(f_tilde, i_tilde)
    f_hat = tf.multiply(f_tilde, tf.add(tf.multiply(f, i_tilde), i_tilde_bar))
    i_hat = tf.multiply(i_tilde, tf.add(tf.multiply(f_tilde, i), f_tilde_bar))
    new_c = tf.add(tf.multiply(f_hat, c), tf.multiply(i_hat, c_hat))
    new_h = tf.multiply(o, tf.tanh(new_c))

    # Register gates and return
    self._gate_dict['master_forget_gate'] = f_tilde
    self._gate_dict['master_input_gate'] = i_tilde
    self._gate_dict['forget_gate'] = f
    self._gate_dict['input_gate'] = i
    self._gate_dict['output_gate'] = o
    return new_h, (new_h, new_c)


  def _get_neurons(self, x, h):
    # Calculate master forget and input gate
    f_tilde = self.neurons(x, h, activation='cumax', scope='master_f')
    i_tilde_bar = self.neurons(x, h, activation='cumax', scope='master_i')
    # Calculate standard forget and input gate
    f = self.neurons(x, h, is_gate=True, scope='f')
    i = self.neurons(x, h, is_gate=True, scope='i')
    o = self.neurons(x, h, is_gate=True, scope='o')
    c_hat = self.neurons(x, h, activation=self._activation, scope='c_hat')

    return f_tilde, i_tilde_bar, f, i, o, c_hat


  def _get_neurons_fast(self, x, h):
    # Calculate net inputs
    net_f_tilde, net_i_tilde_bar, net_f, net_i, net_o, net_c_hat = self.neurons(
      x, h, scope='net_inputs', num_or_size_splits=6)

    # Activate
    cumax = activations.get('cumax')
    f_tilde = cumax(net_f_tilde)
    i_tilde_bar = cumax(net_i_tilde_bar)
    f = tf.nn.sigmoid(net_f)
    i = tf.nn.sigmoid(net_i)
    o = tf.nn.sigmoid(net_o)
    c_hat = self._activation(net_c_hat)

    return f_tilde, i_tilde_bar, f, i, o, c_hat


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from collections import OrderedDict

from tframe import activations
from tframe import checker
from tframe import context
from tframe import initializers
from tframe.nets.rnet import RNet


class ClockworkRNN(RNet):
  """Clockwork RNN Unit, currently can not work with parallel engine
     References:
       [1] https://arxiv.org/abs/1402.3511
  """
  net_name = 'cw_rnn'

  def __init__(
      self,
      state_size,
      periods=None,
      activation='tanh',
      use_bias=True,
      weight_initializer='xavier_uniform',
      bias_initializer='zeros',
      **kwargs):
    """
    :param state_size: State size
    :param periods: a list of integers. If not provided, periods will be set
                    to a default exponential series {2^{i-1}}_{i=0}^{state_size}
    """
    # Call parent's constructor
    RNet.__init__(self, ClockworkRNN.net_name)

    # Attributes
    self._state_size = checker.check_positive_integer(state_size)
    self._periods = self._get_periods(periods, **kwargs)
    self._activation = activations.get(activation, **kwargs)
    self._use_bias = checker.check_type(use_bias, bool)
    self._weight_initializer = initializers.get(weight_initializer)
    self._bias_initializer = initializers.get(bias_initializer)

    # modules = [(start_index, size, period)+]
    self._modules = []
    self._init_modules(**kwargs)

  # region : Properties

  def structure_string(self, detail=True, scale=True):
    return self.net_name + '({})'.format(self._state_size)

  @property
  def init_state(self):
    if self._init_state is not None: return self._init_state
    self._init_state = (self._get_placeholder('h', self._state_size),
                        tf.placeholder(tf.int32, shape=(None, 1), name='clock'))
    return self._init_state

  # endregion : Properties

  # region : Core

  def _link(self, h_clock, x, **kwargs):
    # Sanity check
    self._check_state(h_clock, (self._state_size, 1))
    h, clock = h_clock

    # Execute all modules (TODO: too much of brutal force)
    h_x = tf.concat([h, x], axis=1, name='h_x')
    results = []
    for m in self._modules:
      assert isinstance(m, Module)
      results.append(m.execute(h_x))
    result = tf.concat(results, axis=1)

    # Bias
    bias = None
    if self._use_bias: bias = self._get_bias('bias', self._state_size)

    # Calculate f(h * Wh + x * Wx + b)
    y = self._activation(tf.nn.bias_add(result, bias))

    # Calculate and apply mask
    mask = tf.cast(tf.equal(tf.mod(clock, self._periods), 0), tf.float32)
    # mask = tf.Print(mask, [clock, mask, self._periods], '[Clock, Mask] = ')

    updated = tf.multiply(y, mask)
    new_h = updated + tf.multiply(1.0 - mask, h)

    # Return output and new state
    new_clock = tf.add(clock, 1)
    return new_h, (new_h, new_clock)

  # endregion : Core

  # region : Private

  def _get_periods(self, periods, **kwargs):
    # Get max groups
    max_groups = kwargs.get('max_groups', 7)
    if periods is None:
      periods = []
      i = 0
      for _ in range(self._state_size):
        periods.append(2 ** i)
        i += 1
        if i >= max_groups: i = 0
    else: checker.check_type(periods, int)
    assert len(periods) == self._state_size
    return sorted(periods)

  def _init_modules(self, **kwargs):
    for i, p in enumerate(self._periods):
      if len(self._modules) == 0 or self._modules[-1].period != p:
        self._modules.append(Module(p, i, self))
      else: self._modules[-1].add()

  # endregion : Private

class Module(object):
  def __init__(self, period, start_index, master):
    self.period = checker.check_positive_integer(period)
    self.start_index = checker.check_type(start_index, int)
    self.size = 1
    assert isinstance(master, ClockworkRNN)
    self.master = master

  def add(self):
    self.size += 1

  def execute(self, h_x):
    part = h_x[:, self.start_index:]
    assert isinstance(part, tf.Tensor)
    height = part.shape.as_list()[1]
    W = self.master._get_variable(
      'W{}'.format(self.period), shape=[height, self.size])
    return tf.matmul(part, W)


if __name__ == '__main__':
  cwrnn = ClockworkRNN(state_size=100)
  for p in cwrnn._periods:
    print(p)

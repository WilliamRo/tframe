from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe.nets.rnn_cells.srn import BasicRNNCell


class PRU(BasicRNNCell):
  """Prototypical Recurrent Unit
  Reference: https://www.sciencedirect.com/science/article/pii/S0925231218306283
  """
  net_name = 'pru'

  def _link(self, s, x, **kwargs):
    self._check_state(s)

    # :: F - update memory
    # .. Calculate candidates
    u = self.neurons(x, s, scope='candidates', activation='tanh')
    # .. Calculate update gate
    c = self.neurons(x, s, scope='update_gate', is_gate=True)
    self._gate_dict['update_gate'] = c
    # .. Update memory
    with tf.name_scope('F'):
      new_s = tf.add(tf.multiply(c, s), tf.multiply(tf.subtract(1., c), u))

    # :: G - calculate output
    y = self.neurons(new_s, activation=self._activation, scope='G')

    # Return output and new state
    return y, new_s



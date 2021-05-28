from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe.models.recurrent import Recurrent
from tframe import hub as th

from .node_register import NodeRegister


class RealTimeOptimizer(object):
  """Real-time optimizer for Recurrent Neural Networks
     Here real-time means num_steps = 1 and batch_size = 1

     If the RNet has more than one recurrent layers, say, the
      corresponding hidden states are denoted [h_1, h_2, ...] in order
     the gradient dh_p(t)/dh_q(t-1) is truncated to 0 for p > q

     TODO: This is a BETA version. Pls use under the guidance.
  """
  def __init__(self, rnn, tf_optimizer):
    """Initiate a real-time optimizer
    :param rnn: an instance of tframe recurrent net
    :param tf_optimizer: an instance of tensorflow.train.Optimizer
    """
    # Check th
    # assert th.batch_size == 1 and th.num_steps == 1

    assert isinstance(rnn, Recurrent)
    self._rnn = rnn
    assert isinstance(tf_optimizer, tf.train.Optimizer)
    self._tf_optimizer = tf_optimizer

    # The buffer stores ...
    self._register = None

  # region : Public Methods

  def minimize(self, loss, var_list=None):
    # Sanity check: make sure self._rnn has already been built so the dynamic
    #  nodes and weights can be found
    if not self._rnn.linked:
      raise AssertionError('!! The net `{}` should be linked'.format(
        self._rnn.name))
    # Initialize register
    self._register = NodeRegister(self._rnn)

    # Step 1: compute gradients
    grads_and_vars = self._compute_gradients(loss, var_list=var_list)
    # Step 2: apply gradients
    return self._tf_optimizer.apply_gradients(grads_and_vars)

  # endregion : Public Methods

  # region : Private Methods

  def _compute_gradients(self, loss, var_list=None):
    """Compute gradients using RTRL. The default compute_gradients method of
       the given optimizer will be called. Besides, a Block in the register
       may add additional gradients to its corresponding weight gradients if
       a `` method if provided.

    :param loss: the loss tensor
    :param var_list: variable list (may not be used for now)
    :return: A list of (gradient, variable) pairs as the compute_gradients
              method does in tf.Optimizer
    """
    # Sanity check
    assert isinstance(loss, tf.Tensor)

    # Compute gradients using default method
    assert isinstance(self._register, NodeRegister)
    default_grads_and_vars = self._tf_optimizer.compute_gradients(
      loss, var_list=self._register.default_var_list)

    # Compute gradients using customized method held
    dL_dy = tf.gradients(loss, self._rnn.last_scan_output)[0]
    c_g_n_v, new_buffer = self._register.compute_customized_gradient(dL_dy)
    self._rnn.grad_buffer_slot.plug(new_buffer)

    grads_and_vars = default_grads_and_vars + c_g_n_v
    if th.test_grad:
      _grads_and_vars = self._tf_optimizer.compute_gradients(loss)
      deltas_and_vars = []
      deltas = []
      for _g, _v in _grads_and_vars:
        matches = [g for g, v in grads_and_vars if v is _v]
        assert len(matches) == 1
        g = matches[0]

        delta_name = '_'.join(_v.name.split('/'))
        delta = tf.subtract(g, _g, name='delta_{}'.format(delta_name[:-2]))
        deltas_and_vars.append((delta, _v))
        deltas.append(delta)

      self._rnn.grad_delta_slot.plug(tuple(deltas))

    return grads_and_vars

  # endregion : Private Methods




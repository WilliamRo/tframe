from collections import OrderedDict

import numpy as np

from tframe.advanced.synapspring.springs.spring_base import SpringBase
from tframe import context
from tframe import hub as th
from tframe import tf
from tframe.utils.maths.stat_tools import Statistic


class SynapticIntelligence(SpringBase):
  """This module implements Synaptic Intelligence proposed in
     Zenke, et. al., 2017.
  """

  name = 'CL-REG-SI'

  def __init__(self, model):
    # Call parent's initializer
    super(SynapticIntelligence, self).__init__(model)
    th.monitor_weight_grads = True
    th.monitor_weight_history = True

    self.epsilon = 1e-3

  # region: Properties

  @SpringBase.property()
  def importance_estimates(self):
    od = OrderedDict()
    for v in self.variables:
      shape = v.shape.as_list()
      od[v] = np.zeros(shape, dtype=np.float32)
    return od

  # endregion: Properties

  # region: Implementation of Abstract Methods

  def calculate_loss(self) -> tf.Tensor:
    vars = self.model.var_list
    shadows = self.model.shadows
    assert len(vars) == len(shadows)

    loss_list = []
    for v in vars:
      s = shadows[v]
      omega = self.omegas[v]
      loss_list.append(tf.reduce_sum(omega * tf.square(s - v)))

    return tf.multiply(th.cl_reg_lambda, tf.add_n(loss_list), name=self.name)

  def call_after_each_update(self):
    """Update importance estimates"""

    assert th.monitor_weight_grads
    monitor = context.monitor
    grads = monitor._weight_grad_dict
    weight_history = monitor._weight_history

    for v in self.variables:
      # Calculate w_delta
      s: Statistic = weight_history[v]
      assert len(s._value_list) == 2
      w_delta = s.last_value - s._value_list[0]

      # Get grad at current step
      g = grads[v].last_value

      # Update importance estimates
      w = self.importance_estimates[v]
      w = w - g * w_delta
      self.importance_estimates[v] = w

  def _update_omega(self):
    ops = []
    for v in self.variables:
      w = self.importance_estimates[v]
      shadow = self.model.shadows[v]
      omega_add = w / (tf.square(v - shadow) + self.epsilon)
      ops.append(tf.assign_add(self.omegas[v], omega_add))
    self.model.session.run(ops)

  # endregion: Implementation of Abstract Methods










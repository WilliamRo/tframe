from collections import OrderedDict

import numpy as np

from tframe.advanced.synapspring.springs.spring_base import SpringBase
from tframe import tf
from tframe import hub as th



class SynapticIntelligence(SpringBase):

  def __init__(self, model):
    # Call parent's initializer
    super(SynapticIntelligence, self).__init__(model)

    self.epsilon = 1e-8

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
      loss_list.append(tf.reduce_mean(tf.square(s - v)))

    return tf.multiply(th.cl_reg_lambda, tf.add_n(loss_list), name='cl_reg_l2')

  def call_after_each_update(self):
    """Update importance estimates"""
    # TODO
    pass

  def update_omega_after_training(self):
    ops = []
    for v in self.variables:
      w = self.importance_estimates[v]
      shadow = self.model.shadows[v]
      omega_add = w / (tf.square(v - shadow) + self.epsilon)
      ops.append(tf.assign_add(self.omegas[v], omega_add))
    self.model.session.run(ops)

  # endregion: Implementation of Abstract Methods










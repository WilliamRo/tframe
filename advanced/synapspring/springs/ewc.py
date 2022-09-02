from collections import OrderedDict

import numpy as np

from tframe.advanced.synapspring.springs.spring_base import SpringBase
from tframe import context
from tframe import hub as th
from tframe import tf
from tframe.utils.maths.stat_tools import Statistic


class EWC(SpringBase):
  """This module implements Synaptic Intelligence proposed in
     Kirkpatrick, 2017
  """

  def __init__(self, model):
    # Call parent's initializer
    super(EWC, self).__init__(model)
    th.monitor_weight_grads = True
    th.monitor_weight_history = True

    self.epsilon = 1e-8


  # region: Implementation of Abstract Methods



  # endregion: Implementation of Abstract Methods

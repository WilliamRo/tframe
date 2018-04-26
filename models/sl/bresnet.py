from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import Predictor

from tframe import losses
from tframe import metrics

from tframe.core import with_graph
from tframe.models import Feedforward
from tframe.nets.net import Net



class BResNet(Predictor):
  """Branch-Residual Net...? Fine, it's just a temporary name."""
  def __init__(self, mark=None):
    # Call parent's initializer
    Predictor.__init__(self, mark)
    # Private attributes
    self._output_tensors = []
    self._loss_tensors = []
    self._metric_tensors = []
    self._train_ops = []

  # region : Build

  @with_graph
  def _build(self, loss='cross_entropy', optimizer=None,
             metric=None, metric_is_like_loss=True, metric_name='Metric'):
    Feedforward._build(self)
    

  # endregion : Build

  # region : Train
  # endregion : Train

  # region : Public Methods
  # endregion : Public Methods

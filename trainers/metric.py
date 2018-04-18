from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from tframe import console

from tframe.core import TensorSlot, VariableSlot, SummarySlot


class Metric(TensorSlot):

  def __init__(self, model, name):
    # Call parent's constructor
    super().__init__(model, name)
    #
    self._as_loss = None
    self.symbol = None
    # :: Attribute for take down metric history
    self._metric_logs = [[]]
    self._record = VariableSlot(self._model)
    self._mean_record = VariableSlot(self._model)
    # For hp-tuning
    self._record_summary = SummarySlot(self._model)
    # TODO
    self._memory = 4
    self._trend = []

  # region : Properties

  @property
  def record(self):
    return self._record.fetch()

  @record.setter
  def record(self, value):
    self._record.assign(value)

  @property
  def mean_record(self):
    return self._mean_record.fetch()

  @mean_record.setter
  def mean_record(self, value):
    self._mean_record.assign(value)

  # endregion : Properties

  # region : Public Methods

  def is_better_than(self, metric1, metric2, gap=0):
    assert self._as_loss is not None
    if self._as_loss: return metric1 < metric2 - gap
    else: return metric1 > metric2 + gap

  def end_round(self):
    new_record = False
    assert isinstance(self._metric_logs[-1], list)
    metric_mean = np.mean(self._metric_logs.pop())
    self._metric_logs.append(metric_mean)
    trend = []
    for i in range(min(self._memory, len(self._metric_logs) - 1)):
      hist_mean = self._metric_logs[-(i + 2)]
      assert hist_mean > 0.0
      trend.append((metric_mean - hist_mean) / hist_mean * 100)
    self._trend = trend

    # Update best mean metric
    mean_record = self.mean_record
    if (self._mean_record.never_assigned
        or self.is_better_than(metric_mean, mean_record)):
      self.mean_record = metric_mean
      mean_record = metric_mean
      new_record = True

    # Show metric mean status
    console.supplement('E[metric] = {:.3f}, min(E[metric]) = {:.3f}'.format(
      metric_mean, mean_record))
    self._show_trend()

    # Append log container for new round
    self._metric_logs.append([])

    return new_record


  def add_to_log(self, metric):
    log_for_current_round = self._metric_logs[-1]
    assert isinstance(log_for_current_round, list)
    log_for_current_round.append(metric)

  # endregion : Public Methods

  # region : Methods Overrode

  def plug(self, op, as_loss=True, symbol='metric'):
    self._as_loss = as_loss
    self.symbol = symbol
    super().plug(op)
    self._init_tensors()

  # endregion : Methods Overrode

  # region : Private Methods

  def _init_tensors(self):
    with self._model.graph:
      self._record.plug(tf.Variable(
        initial_value=0.0, trainable=False, name='metric_record'))
      self._mean_record.plug(tf.Variable(
        initial_value=0.0, trainable=False, name='metric_mean_record'))
      self._record_summary.plug(tf.summary.scalar(
        'metric_record_sum', self._record.tensor))

  def _show_trend(self):
    tendency = ''
    if len(self._trend) > 0:
      for i, ratio in enumerate(self._trend):
        if i > 0: tendency += ', '
        tendency += '[{}]{:.1f}%'.format(i + 1, ratio)
    if tendency != '': console.supplement(tendency, level=2)

  # endregion : Private Methods

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import console
from tframe.core import TensorSlot, VariableSlot, SummarySlot


class Metric(TensorSlot):

  def __init__(self, model, name='metric'):
    # Call parent's constructor
    super().__init__(model, name)
    #
    self._as_loss = None
    self.symbol = None
    self._record_round = 0
    # :: Attribute for take down metric history
    self._metric_logs = [[]]
    self._record = VariableSlot(self._model)
    self._mean_record = VariableSlot(self._model)
    # For hp-tuning
    self._record_summary = SummarySlot(self._model)
    # TODO
    self.memory = 4
    self._trend = []

  # region : Properties

  @property
  def like_loss(self):
    return self._as_loss

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

  @property
  def trend_is_promising(self):
    if len(self._trend) == 0: return None
    if self._as_loss: return all(np.array(self._trend) < 0)
    else: return all(np.array(self._trend) > 0)

  @property
  def metric_mean_history_str(self):
    history = 'History of E(metric): '
    logs = [l for l in self._metric_logs if np.isscalar(l)]
    for i, l in enumerate(logs):
      history += '[{}]{:.3f} -> '.format(i + 1, l)
    return history

  # endregion : Properties

  # region : Public Methods

  def get_idle_rounds(self, rnd):
    return rnd - self._record_round

  def is_better_than(self, metric1, metric2, gap=0):
    assert self._as_loss is not None
    if self._as_loss: return metric1 < metric2 - gap
    else: return metric1 > metric2 + gap

  def end_round(self, rnd):
    new_record = False
    assert isinstance(self._metric_logs[-1], list)
    metric_mean = np.mean(self._metric_logs.pop())
    self._metric_logs.append(metric_mean)
    trend = []
    for i in range(min(self.memory, len(self._metric_logs) - 1)):
      hist_mean = self._metric_logs[-(i + 2)]
      assert hist_mean >= 0.0
      trend.append((metric_mean - hist_mean) / hist_mean * 100)
    # trend = [re(rnd-1), re(rnd-2), ..., re(rnd-memory)]
    self._trend = trend

    # Update best mean metric
    mean_record = self.mean_record
    if (self._mean_record.never_assigned
        or self.is_better_than(metric_mean, mean_record)):
      self.mean_record = metric_mean
      mean_record = metric_mean
      new_record = True

    # Show metric mean status
    # TODO: console access should be somehow controlled
    token = 'min' if self._as_loss else 'max'
    console.supplement('E[metric] = {:.3f}, {}(E[metric]) = {:.3f}'.format(
      metric_mean, token, mean_record))
    self._show_trend()

    # Show record
    console.supplement(
      '[Best {:.3f}] {} rounds since last record appears.'.format(
        self.record, self.get_idle_rounds(rnd)))

    # Append log container for new round
    self._metric_logs.append([])

    return new_record

  def take_down(self, metric, rnd, gap=0):
    new_record = False
    # Add new metric to log
    self._add_to_log(metric)

    # Update metric record
    if (self._record.never_assigned or
        self.is_better_than(metric, self.record, gap=gap)):
      self._record_round = rnd
      self.record = metric
      new_record = True

    return new_record

  def write_record_summary(self):
    self._model.agent.write_summary(self._record_summary.run())

  # endregion : Public Methods

  # region : Methods Overrides

  def plug(self, op, as_loss=True, symbol='metric'):
    self._as_loss = as_loss
    self.symbol = symbol
    self.name = symbol
    super().plug(op)
    self._init_tensors()

  # endregion : Methods Overrides

  # region : Private Methods

  def _init_tensors(self):
    with self._model.graph.as_default():
      self._record.plug(tf.Variable(
        initial_value=-1.0, trainable=False, name='metric_record'))
      self._mean_record.plug(tf.Variable(
        initial_value=-1.0, trainable=False, name='metric_mean_record'))
      self._record_summary.plug(tf.summary.scalar(
        'metric_record_sum', self._record.tensor))

  def _show_trend(self):
    tendency = ''
    if len(self._trend) > 0:
      for i, ratio in enumerate(self._trend):
        if i > 0: tendency += ', '
        tendency += '[{}]{:.1f}%'.format(i + 1, ratio)
    if tendency != '': console.supplement(tendency, level=2)

  def _add_to_log(self, metric):
    log_for_current_round = self._metric_logs[-1]
    assert isinstance(log_for_current_round, list)
    log_for_current_round.append(metric)

  # endregion : Private Methods

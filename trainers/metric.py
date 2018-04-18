from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

import tframe as tfr
from tframe import console


class Metric(tfr.core.Slot):

  def __init__(self, model, name):
    # Call parent's constructor
    super().__init__(model, name)
    #
    self._as_loss = None
    self.symbol = None
    # :: Attribute for take down metric history
    self._metric_logs = [[]]
    self._best_metric = None
    self._no_record = True
    self._best_metric_mean = None
    self._no_mean_record = True
    # For hp-tuning
    self._best_metric_summary = None
    # TODO
    self._memory = 4
    self._trend = []

  # region : Properties

  @property
  def record(self):
    return self.fetch()

  @record.setter
  def record(self, value):
    assert self.activated
    with self._model.graph.as_default():
      self._model.session.run(tf.assign(self._best_metric, value))
    self._no_record = False

  @property
  def mean_record(self):
    assert self.activated
    with self._model.graph.as_default():
      return self._model.session.run(self._best_metric_mean)

  @mean_record.setter
  def mean_record(self, value):
    assert self.activated
    with self._model.graph.as_default():
      self._model.session.run(tf.assign(self._best_metric_mean, value))
    self._no_mean_record = False

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
    if self._no_mean_record or self.is_better_than(metric_mean, mean_record):
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
      self._best_metric = tf.Variable(
        initial_value=0.0, trainable=False, name='best_metric')
      self._best_metric_mean = tf.Variable(
        initial_value=0.0, trainable=False, name='best_metric_mean')
      self._best_metric_summary = tf.summary.scalar(
        'best_metric_sum', self._best_metric)

  def _show_trend(self):
    tendency = ''
    if len(self._trend) > 0:
      for i, ratio in enumerate(self._trend):
        if i > 0: tendency += ', '
        tendency += '[{}]{:.1f}%'.format(i + 1, ratio)
    if tendency != '': console.supplement(tendency, level=2)

  # endregion : Private Methods

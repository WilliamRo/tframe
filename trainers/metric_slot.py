from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe import console
from tframe.core import TensorSlot, VariableSlot, SummarySlot


class MetricSlot(TensorSlot):

  def __init__(self, model, name='metric', post_processor=None):
    # Call parent's constructor
    super().__init__(model, name)
    #
    self.symbol = None
    self._record_round = 0
    self._record_counter = 0
    # :: Attribute for take down metric history
    self._metric_logs = [[]]
    self._record = VariableSlot(self._model)
    self._mean_record = VariableSlot(self._model)
    # For hp-tuning
    self._record_summary = SummarySlot(self._model)
    # TODO
    self.memory = 4
    self._trend = []

    # Bayesian booster
    self.improvement = 0

    self.post_processor = post_processor

  # region : Properties

  @property
  def lower_is_better(self):
    return self.quantity_definition.lower_is_better

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
    if self.lower_is_better: return all(np.array(self._trend) < 0)
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

  def set_record_round(self, rnd):
    self._record_round = rnd

  def set_record_counter(self, counter):
    self._record_counter = counter

  def get_idle_rounds(self, rnd):
    return rnd - self._record_round

  def get_idle_counts(self, counter):
    return counter - self._record_counter

  def is_better_than(self, metric1, metric2, gap=0):
    assert self.quantity_definition is not None
    if self.lower_is_better: return metric1 < metric2 - gap
    else: return metric1 > metric2 + gap

  def end_round(self, rnd):
    new_record = False
    assert isinstance(self._metric_logs[-1], list)
    if len(self._metric_logs[-1]) == 0: return new_record

    current_metrics = self._metric_logs.pop()
    metric_mean = np.mean(current_metrics)
    self._metric_logs.append(metric_mean)

    trend = []
    for i in range(min(self.memory, len(self._metric_logs) - 1)):
      hist_mean = self._metric_logs[-(i + 2)]
      # assert hist_mean >= 0.0  # Metric can be negtive TODO X
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
    token = 'min' if self.lower_is_better else 'max'
    console.supplement('E[{}] = {:.3f}, {}(E[{}]) = {:.3f}'.format(
      self.symbol, metric_mean, token, self.symbol, mean_record))
    self._show_trend()

    # Show record
    console.supplement(
      '[Best {:.3f}] {} rounds since last record appears.'.format(
        self.record, self.get_idle_rounds(rnd)))

    # Append log container for new round
    self._metric_logs.append([])

    return new_record

  def take_down(self, metric, rnd, counter, gap=0):
    new_record = False
    # Add new metric to log
    self._add_to_log(metric)

    # Update metric record
    if (self._record.never_assigned or
        self.is_better_than(metric, self.record, gap=gap)):
      # Accumulate improvement
      if not self._record.never_assigned:
        self.improvement += abs(metric - self.record)

      self._record_round = rnd
      self._record_counter = counter
      self.record = metric
      new_record = True

    return new_record

  def write_record_summary(self):
    self._model.agent.write_summary(self._record_summary.run())

  # endregion : Public Methods

  # region : Methods Overrides

  def plug(self, op, symbol='metric', quantity_def=None):
    """Called in the building methods of a model"""
    self.symbol = symbol
    super().plug(op, quantity_def=quantity_def)
    self._init_tensors()

  # endregion : Methods Overrides

  # region : Private Methods

  def _init_tensors(self):
    with self._model.graph.as_default():
      self._record.plug(tf.Variable(
        initial_value=-1.0, trainable=False,
        name='{}_record'.format(self.name)))
      self._mean_record.plug(tf.Variable(
        initial_value=-1.0, trainable=False,
        name='{}_mean_record'.format(self.name)))
      self._record_summary.plug(tf.summary.scalar(
        '{}_record_sum'.format(self.name), self._record.tensor))

      # Put metric tensor into 'do not save' list if required
      from tframe import hub
      if not hub.save_records:
        from tframe import pedia
        tf.add_to_collection(pedia.do_not_save, self._record._op)
        tf.add_to_collection(pedia.do_not_save, self._mean_record._op)


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

  # region : Overriding

  def __str__(self): return self.symbol

  # endregion : Overriding

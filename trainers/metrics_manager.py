from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np
from tframe import tf

from tframe import checker
from tframe import console, context
from tframe import hub
from tframe import metrics
from tframe.core.quantity import Quantity
from tframe.utils.maths.stat_tools import Statistic
from tframe.trainers.metric_slot import MetricSlot
from tframe.data.dataset import DataSet


class MetricsManager(object):
  """A MetricsManager managers a list of metrics.
     A manager knows Model and Trainer"""
  def __init__(self, model):
    self.model = model
    self.trainer = None

    self.metrics = []
    self.stats_dict = OrderedDict()

    self.note = OrderedDict()

    self._eval_metric_slot = None

    self.resurrected = False
    self.rar0 = None
    self._RAR = OrderedDict()
    for i in reversed(range(hub.lives)): self._RAR[i] = []

  # region : Properties

  @property
  def early_stop_slot(self):
    metric = self.metrics[0]
    assert isinstance(metric, MetricSlot)
    return metric

  @property
  def early_stop_criterion(self):
    return self.early_stop_slot.record

  @property
  def eval_slot(self):
    if self._eval_metric_slot is None: return self.early_stop_slot
    else: return self._eval_metric_slot

  @property
  def has_metric(self):
    return len(self.metrics) > 0

  @property
  def latest_stats_dict(self):
    assert len(self.stats_dict) > 0
    stats_dict = OrderedDict()
    for data_set, slot_stat in self.stats_dict.items():
      assert isinstance(data_set, DataSet)
      assert isinstance(slot_stat, OrderedDict)
      scalar_dict = OrderedDict()
      for slot, stat in slot_stat.items():
        assert isinstance(slot, MetricSlot) and isinstance(stat, Statistic)
        scalar_dict[slot] = stat.last_value
        
      stats_dict[data_set] = scalar_dict
    return stats_dict

  @property
  def th(self): return self.trainer.th

  @property
  def ready_for_note_taking(self):
    return len(self.stats_dict) > 0

  @property
  def RAR_string(self):
    """e.g. 0.676 => L1(2)->0.678 => L0(3->0.680)"""
    if self.rar0 is None: return 'No RAR info'
    rs = ' => '.join([
      'L{}({}){}'.format(lv, len(scalars), '->{:.3f}'.format(scalars[-1])
      if len(scalars) > 0 else '') for lv, scalars in self._RAR.items()])
    return '{:.3f} => '.format(self.rar0) + rs

  # endregion : Properties

  # region : Public Methods

  def initialize(
      self, metric_list, last_only, target_tensor, output_tensor, **kwargs):
    # Sanity check
    if not isinstance(metric_list, (tuple, list)): metric_list = [metric_list]
    # if isinstance(metric_list, str): metric_list = [metric_list] TODO: X
    checker.check_type([target_tensor, output_tensor], tf.Tensor)

    for metric in metric_list:
      assert isinstance(metric, (str, Quantity))
      if isinstance(metric, str): metric = metric.lower()

      # Get quality
      if metric == 'loss':
        quantity = self.model.loss_quantity
        tensor = quantity.quantity
      else:
        # Initiate a new Quantity
        quantity = metrics.get(metric, last_only=last_only, **kwargs)
        tensor = quantity(target_tensor, output_tensor)

      # Create a metric_slot and plug tensor in
      # name = metric if isinstance(metric, str) else quantity.name
      name = quantity.name
      metric_slot = MetricSlot(self.model, name=name,
                               post_processor=quantity.post_processor)
      metric_slot.plug(tensor, quantity.name, quantity_def=quantity)

      # Append metric slot to metrics list
      self.metrics.append(metric_slot)

      # Add metric slot to validate_group
      self.model.validate_group.add(metric_slot)

      # TODO to be deprecated
      if hub.summary:
        from tframe import pedia
        tf.add_to_collection(
          pedia.validation_summaries,
          tf.summary.scalar(quantity.name + '_sum', tensor))

  def reset_records(self):
    """Reset records. Mainly used by pruner."""
    for slot in self.metrics:
      assert isinstance(slot, MetricSlot)
      slot.record = -1
      slot.mean_record = -1

  def get_slot_by_name(self, name):
    result = [s for s in self.metrics
              if s.name == name or s.symbol.lower() == name]
    if len(result) == 0:
      raise ValueError('!! metrics_manager can not find `{}`'.format(name))
    return result[0]

  def register_eval_slot(self, key):
    """Slot used for evaluating model may be different from early stop
       criterion. For example in image classification tasks where the early
       stop criterion is loss but the evaluation metric is accuracy. """
    assert isinstance(key, str)
    self._eval_metric_slot = self.get_slot_by_name(key)

  # endregion : Public Methods

  # region : Statistics

  def idle_counter(self, slot, rnd):
    assert isinstance(slot, MetricSlot)
    if self.trainer.is_online:
      return slot.get_idle_counts(self.trainer.counter)
    else: return slot.get_idle_rounds(rnd)

  def record_stats_on_dataset(
      self, data_set, slot_scalar_dict, take_down_on_slot=False, rnd=None):
    """
    Currently stats are taken down on instances of class Statistic to
    store metrics on different data set.

    :param data_set: a tframe DataSet
    :param slot_scalar_dict: a dictionary returned by model.validate_model
    :param take_down_on_slot: whether to record stats on metric_slots,
                              usually set to True if data_set is val_set
    :param rnd: if take_down_on_slot, rnd must be provided
    """
    # Sanity check
    assert isinstance(data_set, DataSet)
    assert isinstance(slot_scalar_dict, dict)

    # Initialize an OrderedDict for data_set if necessary
    if data_set not in self.stats_dict.keys():
      self.stats_dict[data_set] = OrderedDict()

    od = self.stats_dict[data_set]
    flag = False
    assert isinstance(od, OrderedDict)
    for slot, scalar in slot_scalar_dict.items():
      assert isinstance(slot, MetricSlot)
      # Initiate a Statistic for slot on data_set if necessary
      if slot not in od.keys(): od[slot] = Statistic(max_length=2)
      stat = od[slot]
      assert isinstance(stat, Statistic)
      # Record
      stat.record(scalar)
      # Take down if necessary
      if take_down_on_slot:
        assert rnd is not None
        new_record = slot.take_down(scalar, rnd, self.model.counter,
                                    hub.record_gap)
        # Take note for later print
        note_key = (data_set, slot)
        if new_record:
          self.note[note_key] = '<New Record>'
          if slot is self.early_stop_slot:
            flag = True
            if self.resurrected: self._record_after_resurrection(scalar)
        else:
          idle = self.idle_counter(slot, rnd)
          if hub.early_stop and slot is self.early_stop_slot:
            idle_info = 'Patience {}/{}'.format(idle, self.th.patience)
          else: idle_info = 'Idle: {}'.format(idle)
          suffix = '(Best: {}, {})'.format(
            hub.decimal_str(slot.record, hub.val_decimals), idle_info)
          self.note[note_key] = suffix

    return flag

  def print_latest_stats(self, prompt='[Validate]', decimals=3):
    assert isinstance(prompt, str)
    stats_dict = self.latest_stats_dict
    assert isinstance(stats_dict, OrderedDict)
    for data_set, scalar_dict in stats_dict.items():
      assert isinstance(data_set, DataSet)
      assert isinstance(scalar_dict, OrderedDict)
      console.show_status('On {}'.format(data_set.name), prompt)
      for slot, value in scalar_dict.items():
        info = ('{} = {:.' + str(decimals) + 'f}').format(slot.symbol, value)
        # Look up for suffix in note
        note_key = (data_set, slot)
        if note_key in self.note.keys():
          info = '{} {}'.format(info, self.note[note_key])
          if slot is self.early_stop_slot: info = '(*) ' + info
        # Supplement
        console.supplement(info)
    # Recover progress bar if necessary
    self.trainer.recover_progress()

  def update_scalar_dict(self, scalar_dict):
    # Sanity check
    assert isinstance(scalar_dict, dict)
    # Get stats dict
    stats_dict = self.latest_stats_dict
    assert isinstance(stats_dict, OrderedDict)
    for data_set, sd in stats_dict.items():
      assert isinstance(data_set, DataSet) and isinstance(sd, OrderedDict)
      data_prefix = data_set.name[:8]
      for slot, value in sd.items():
        scalar_dict['{} {}'.format(data_prefix, slot.symbol)] = value

  # endregion : Statistics

  # region : Private Methods

  def _record_after_resurrection(self, scalar):
    lives = self.trainer._lives
    assert isinstance(self._RAR[lives], list)
    self._RAR[lives].append(scalar)

  # endregion : Private Methods


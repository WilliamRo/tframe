from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time

import numpy as np
import tensorflow as tf

import tframe as tfr

from tframe import TFData
from tframe import config
from tframe import console
from tframe import pedia
from tframe.utils import imtool

from tframe.enums import InputTypes
from tframe.core import with_graph
from tframe.core import Agent
from tframe.core import TensorSlot, SummarySlot, OperationSlot
from tframe.core import Group

from tframe.nets.net import Net

from tframe.trainers.metric import Metric
from tframe.trainers.trainer import Trainer, TrainerHub
from tframe.trainers.smartrainer import SmartTrainer, SmartTrainerHub


class Model(object):
  """
  Base class of [all?] kinds of models built on TensorFlow
  """
  model_name = 'default'

  def __init__(self, mark=None):
    # Model mark usually helps to decide the folder name
    self.mark = config.mark or mark
    assert mark is not None

    # Each model has an agent to deal with some tensorflow stuff
    self.agent = Agent(self)

    # Define slots
    self._outputs = TensorSlot(self)

    self._loss = TensorSlot(self, 'Train loss')
    self._train_step = OperationSlot(self)
    self._merged_summary = SummarySlot(self)
    self._update_group = Group(
      self, self._loss, self._train_step, self._merged_summary,
      name='Update-group')

    self._metric = Metric(self, 'metric')
    self._validation_summary = SummarySlot(self)
    self._validate_group = Group(
      self, self._metric, self._validation_summary, name='Validate-group')

    # Private attributes
    self._optimizer = None
    self._built = False

    # Public attributes
    self.counter = None

  # region : Properties

  # region : Accessor

  @property
  def graph(self):
    return self.agent.graph

  @property
  def session(self):
    return self.agent.session

  @property
  def metric(self):
    if self._metric is not None:
      assert isinstance(self._metric, Metric)
    return self._metric

  @property
  def built(self):
    assert isinstance(self._built, bool)
    return self._built

  # endregion : Accessor

  # region : Properties to be overrode

  @property
  def description(self):
    return 'No description'

  @property
  def input_type(self):
    return InputTypes.BATCH

  # endregion : Properties to be overrode

  # endregion : Properties

  # region : Building

  def build(self, *args, **kwargs):
    """Abstract method, must be implemented in different models"""
    raise  NotImplementedError('!! build method not implemented')

  @with_graph
  def _define_train_step(self, optimizer=None, var_list=None):
    if not self._loss.activated:
      raise AssertionError('!! loss has not been activated yet')
    with tf.name_scope('Optimizer'):
      if optimizer is None: optimizer = tf.train.AdamOptimizer(1e-4)
      self._optimizer = optimizer
      self._train_step.plug(
        optimizer.minimize(self._loss.op, var_list=var_list))

  # endregion : Building

  # region : Training

  def pretrain(self, **kwargs):
    """Method run in early training process, should be overrode"""
    pass

  # TODO
  def _apply_smart_train(self):
    # ...
    # Tune learning rate TODO: smart train will only be applied here
    if len(self._metric_log) >= memory + 1 and self._train_status['metric_on']:
      if all(np.array(history) < 0) and self._train_status['bad_apples'] > 0:
        self._train_status['bad_apples'] -= 1
      console.supplement('{} bad apples found'.format(
        self._train_status['bad_apples']))
      if self._train_status['bad_apples'] > memory and FLAGS.smart_train:
        self._tune_lr(lr_decay)
        self._train_status['bad_apples'] = 0
        if not FLAGS.save_best:
          console.show_status('save_best option has been turned on')
        FLAGS.save_best = True

    return save_flag

  # TODO
  @with_graph
  def train(self, epoch=1, batch_size=128, training_set=None,
            validation_set=None, print_cycle=0, snapshot_cycle=0,
            snapshot_function=None, probe=None, **kwargs):
    """"""
    trainer_class = SmartTrainer if config.smart_train else Trainer
    trainer = trainer_class(self, training_set=training_set,
                            validation_set=validation_set,
                            snapshot=snapshot_function, probe=probe)
    trainer.train(epoch=epoch, batch_size=batch_size, print_cycle=print_cycle,
                  snapshot_cycle=snapshot_cycle, **kwargs)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Begin iteration
    with self._session.as_default():
      for epc in range(epoch):
        # Add a new list to metric log if smart_train is on
        # Record epoch start time
        while True:
          # Print status
          if print_cycle > 0 and np.mod(self.counter - 1, print_cycle) == 0:
            loss_dict, new_record = self._update_loss_dict(loss_dict, probe)
            self._print_progress(epc, start_time, loss_dict,
                                 data_batch=data_batch)
            if new_record:
              self._last_epoch = epc
              if FLAGS.save_best and epc + 1 >= FLAGS.dont_save_until:
                if FLAGS.save_model:
                  self._save(self.counter)
                  self._inter_cut('[New Record] Model saved')

          # Snapshot
          if (FLAGS.snapshot and snapshot_cycle > 0
              and np.mod(self.counter - 1, snapshot_cycle) == 0):
            self._snapshot()
          # Check flag: break

        # End of epoch
        since_last = epc - self._last_epoch
        if since_last == 0: self._train_status['bad_apples'] = 0
        else: self._train_status['bad_apples'] += 1
        save_flag = self._apply_smart_train() if FLAGS.smart_train else True
        if self._train_status['metric_on']:
          best_metric = self._session.run(self._best_metric)
          console.supplement(
            '[Best {:.3f}] {} epochs since last record appears.'.format(
            best_metric, since_last))

        if not FLAGS.save_best and save_flag and FLAGS.save_model:
          self._save(self._counter)
          console.show_status('Model saved')
        elif since_last >= epoch_tol: break_flag = True

        # Early stop if break flag is true
        if break_flag: break

    # End training
    if FLAGS.progress_bar: console.clear_line()

    # Write HP-tuning metric
    if FLAGS.hpt:
      summary = self._session.run(self._best_metric_sum)
      self._summary_writer.add_summary(summary, self.counter)

    if FLAGS.summary or FLAGS.hpt: self._summary_writer.flush()
    # TODO: shutdown at an appropriate time
    # self.shutdown()

  def update_model(self, data_batch, **kwargs):
    """Default model updating method, should be overrode"""
    feed_dict = self._get_default_feed_dict(data_batch, is_training=True)
    return self._update_group.run(feed_dict)

  def validate_model(self, validation_set, **kwargs):
    assert isinstance(validation_set, TFData)
    feed_dict = self._get_default_feed_dict(validation_set, is_training=False)
    return self._validate_group.run(feed_dict)

  # endregion : Training

  # region : Static Methods

  @staticmethod
  def show_building_info(**kwargs):
    console.show_status('Model built successfully:')
    for k, v in kwargs.items():
      assert isinstance(v, Net)
      console.supplement('{}: {}'.format(k, v.structure_string()))

  # endregion : Static Methods

  # region : Public Methods

  def shutdown(self):
    self.agent.shutdown()

  def launch_model(self, overwrite=False):
    return self.agent.launch_model(overwrite)

  # endregion : Public Methods

  # region : Private Methods

  def _tune_lr(self, alpha):
    assert np.isscalar(alpha) and 0 < alpha < 1
    lr, new_lr = None, None
    if isinstance(self._optimizer, tf.train.AdamOptimizer):
      lr = self._optimizer._lr
      new_lr = lr * alpha
      self._optimizer._lr = new_lr
      self._lr_t = tf.constant(new_lr)
    elif isinstance(self._optimizer, tf.train.GradientDescentOptimizer):
      lr = self._optimizer._learning_rate
      new_lr = lr * alpha
      self._optimizer._learning_rate = new_lr
    else: return
    # Show status
    console.show_status(
      'learning rate updated: {:.7f} => {:.7f}'.format(lr, new_lr))

  @with_graph
  def _get_default_feed_dict(self, batch, is_training):
    feed_dict = {}
    for tensor in tf.get_collection(pedia.default_feed_dict):
      if 'input' in tensor.name.lower():
        feed_dict[tensor] = batch[pedia.features]
      elif 'target' in tensor.name:
        # TODO: when predict without outputing loss ...
        if batch.targets is not None: feed_dict[tensor] = batch.targets

    feed_dict.update(self.agent.get_status_feed_dict(is_training))

    return feed_dict

  # endregion : Private Methods

  """For some reason, do not remove this line"""


if __name__ == '__main__':
  console.show_status('__main__')





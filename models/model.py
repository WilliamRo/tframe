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

from tframe.core import with_graph

from tframe.nets.net import Net

from tframe.utils.local import check_path
from tframe.utils.local import clear_paths
from tframe.utils.local import load_checkpoint
from tframe.utils.local import save_checkpoint
from tframe.utils.local import write_file

from tframe.trainers.trainer import Trainer, TrainerHub
from tframe.trainers.smartrainer import SmartTrainer, SmartTrainerHub


class Model(object):
  """
  Base class of some kinds of models
  """
  # TODO: Model should be generalized further. Subclasses for sl/usl and rl
  #       should be inherited from this separately

  model_name = 'default'

  def __init__(self, mark=None):
    self.mark = config.mark or mark
    assert mark is not None

    self._metric = None
    self._merged_summary = None
    self._print_summary = None

    self._session = None
    self._summary_writer = None
    self._saver = None

    self._outputs = None
    self._loss = None
    self._train_step = None
    self._optimizer = None

    self._counter = None

    self._snapshot_function = None

    self._built = False

    self._last_epoch = 0

    # Each model is bound to a unique graph
    self._graph = tf.Graph()
    with self._graph.as_default():
      with tf.name_scope('misc'):
        self._best_metric = tf.Variable(
          initial_value=-1.0, trainable=False, name='best_metric')
        self._best_metric_sum = tf.summary.scalar(
          'best_metric_sum', self._best_metric)
        self._best_mean_metric = tf.Variable(
          initial_value=-1.0, trainable=False, name='best_mean_metric')
        self._is_training = tf.placeholder(dtype=tf.bool, name=pedia.is_training)
    # When linking batch-norm layer (and dropout layer),
    #   this placeholder will be got from default graph
    self._graph.is_training = self._is_training
    # Record current graph
    tfr.current_graph = self._graph

  # region : Properties

  # region : Accessor

  @property
  def graph(self):
    return self._graph

  @property
  def session(self):
    return self._session

  @property
  def metric(self):
    return self._metric

  @property
  def built(self):
    return self._built

  # endregion : Accessor

  # region : Paths

  @property
  def log_dir(self):
    return check_path(config.job_dir, config.record_dir,
                      config.log_folder_name, self.mark)

  @property
  def ckpt_dir(self):
    return check_path(config.job_dir, config.record_dir,
                      config.ckpt_folder_name, self.mark)

  @property
  def snapshot_dir(self):
    return check_path(config.job_dir, config.record_dir,
                      config.snapshot_folder_name, self.mark)

  # endregion : Paths

  # region : Properties to be overrode

  @property
  def description(self):
    return 'No description'

  # endregion : Properties to be overrode

  # endregion : Properties

  # region : Building

  def build(self, *args, **kwargs):
    """Abstract method, must be implemented in different models"""
    raise  NotImplementedError('build method not implemented')

  @with_graph
  def _define_train_step(self, optimizer=None, var_list=None):
    if self._loss is None:
      raise ValueError('loss has not been defined yet')
    with tf.name_scope('Optimizer'):
      if optimizer is None: optimizer = tf.train.AdamOptimizer(1e-4)
      self._optimizer = optimizer

      self._train_step = optimizer.minimize(loss=self._loss, var_list=var_list)

  # endregion : Building

  # region : Training

  def pretrain(self, **kwargs):
    """Method run in early training process, should be overrode"""
    pass

  def _apply_smart_train(self):
    memory = 4
    lr_decay = FLAGS.lr_decay

    save_flag = False

    # At the end of each epoch, analyze metric log
    assert isinstance(self._metric_log[-1], list)
    metric_mean = np.mean(self._metric_log.pop())
    self._metric_log.append(metric_mean)
    history = []
    for i in range(min(memory, len(self._metric_log) - 1)):
      hist_mean = self._metric_log[-(i + 2)]
      assert hist_mean > 0
      history.append((metric_mean - hist_mean) / hist_mean * 100)

    # Refresh best mean metric
    best_mean_metric = self._session.run(self._best_mean_metric)
    if metric_mean < best_mean_metric or best_mean_metric < 0:
      save_flag = True
      self._session.run(tf.assign(self._best_mean_metric, metric_mean))
      best_mean_metric = metric_mean

    # Show status
    tendency = ''
    if len(history) > 0:
      for i, ratio in enumerate(history):
        if i > 0: tendency += ', '
        tendency += '[{}]{:.1f}%'.format(i + 1, ratio)
    console.supplement('E[metric] = {:.3f}, min(E[metric]) = {:.3f}'.format(
      metric_mean, best_mean_metric))
    if tendency != '': console.supplement(tendency, level=2)

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
        console.section('Epoch {}'.format(epc + 1))
        # Add a new list to metric log if smart_train is on
        if self._train_status['metric_on']: self._metric_log.append([])
        # Record epoch start time
        start_time = time.time()
        while True:
          # Get data batch
          data_batch, end_epoch_flag = self._training_set.next_batch(
            shuffle=FLAGS.shuffle and FLAGS.train)
          # Increase counter, counter may be used in _update_model
          self._counter += 1
          # Update model
          loss_dict = self._update_model(data_batch, **kwargs)
          # Print status
          if print_cycle > 0 and np.mod(self._counter - 1, print_cycle) == 0:
            loss_dict, new_record = self._update_loss_dict(loss_dict, probe)
            self._print_progress(epc, start_time, loss_dict,
                                 data_batch=data_batch)
            if new_record:
              self._last_epoch = epc
              if FLAGS.save_best and epc + 1 >= FLAGS.dont_save_until:
                if FLAGS.save_model:
                  self._save(self._counter)
                  self._inter_cut('[New Record] Model saved')

          # Snapshot
          if (FLAGS.snapshot and snapshot_cycle > 0
              and np.mod(self._counter - 1, snapshot_cycle) == 0):
            self._snapshot()
          # Check flag
          if end_epoch_flag:
            if FLAGS.progress_bar: console.clear_line()
            console.show_status('End of epoch. Elapsed time is ' 
                                '{:.1f} secs'.format(time.time() - start_time))
            break

        # End of epoch
        break_flag = False
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
      self._summary_writer.add_summary(summary, self._counter)

    if FLAGS.summary or FLAGS.hpt: self._summary_writer.flush()
    # TODO: shutdown at an appropriate time
    # self.shutdown()

  def _update_loss_dict(self, loss_dict, probe):
    # Update loss dictionary by adding metric (and probe) information
    if self._metric is None or self._validation_set is None:
      return loss_dict, False

    new_record = False

    # Calculate metric
    # assert isinstance(self._metric, tf.Tensor)
    feed_dict = self._get_default_feed_dict(
      self._validation_set, is_training=False)

    # _print_summary is written with print cycle
    if self._print_summary is None or not FLAGS.summary:
      metric, best_metric = self._session.run(
        [self._metric.tensor, self._best_metric], feed_dict=feed_dict)
    else:
      metric, summary, best_metric = self._session.run(
        [self._metric.tensor, self._print_summary, self._best_metric],
        feed_dict=feed_dict)
      assert isinstance(self._summary_writer, tf.summary.FileWriter)
      self._summary_writer.add_summary(summary, self._counter)

    # Add metric info to loss dictionary for printing
    loss_dict.update({pedia.memo[pedia.metric_name]: metric})

    # Add metric information to log maintained by model(self)
    if self._train_status['metric_on']:
      assert isinstance(self._metric_log[-1], list)
      self._metric_log[-1].append(metric)

    # Try to add probing information
    if probe is not None:
      assert callable(probe)
      loss_dict.update({'Probe': probe(self)})

    # TODO: Save best
    delta = best_metric - metric
    if pedia.memo[pedia.metric_name] == pedia.Accuracy: delta = -delta
    if delta > 2e-4 or best_metric < 0:
      new_record = True
      self._session.run(tf.assign(self._best_metric, metric))

    return loss_dict, new_record

  def _update_model(self, data_batch, **kwargs):
    """Default model updating method, should be overrode"""
    feed_dict = self._get_default_feed_dict(data_batch, is_training=True)

    if FLAGS.summary:
      summary, loss, _ = self._session.run(
        [self._merged_summary, self._loss, self._train_step],
        feed_dict=feed_dict)

      assert isinstance(self._summary_writer, tf.summary.FileWriter)
      self._summary_writer.add_summary(summary, self._counter)
    else:
      loss, _ = self._session.run(
        [self._loss, self._train_step], feed_dict=feed_dict)

    loss_dict = collections.OrderedDict()
    loss_dict['Train loss'] = loss
    return loss_dict

  def _print_progress(self, epc, start_time, loss_dict, **kwargs):
    # generate loss string
    loss_strings = ['{} = {:.3f}'.format(k, loss_dict[k])
                    for k in loss_dict.keys()]
    loss_string = ', '.join(loss_strings)

    total_epoch = self._counter / self._training_set.batches_per_epoch
    self._inter_cut(
      'Epoch {} [{:.1f} Total] {}'.format(epc + 1, total_epoch, loss_string),
      start_time=start_time)

  def _snapshot(self):
    if self._snapshot_function is None:
      return

    fig = self._snapshot_function(self)
    epcs = 1.0 * self._counter / self._training_set.batches_per_epoch
    filename = 'train_{:.2f}_epcs.png'.format(epcs)
    imtool.save_plt(fig, "{}/{}".format(self.snapshot_dir))

    self._inter_cut("[Snapshot] images saved to '{}'".format(filename))

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
    if config.summary or config.hp_tuning:
      self._summary_writer.close()
    self._session.close()

  def launch_model(self, overwrite=False):
    config.smooth_out_conflicts()
    # Before launch session, do some cleaning work
    if overwrite and config.overwrite:
      paths = []
      if config.summary: paths.append(self.log_dir)
      if config.save_model: paths.append(self.ckpt_dir)
      if config.snapshot: paths.append(self.snapshot_dir)
      clear_paths(paths)

    # Launch session on self.graph
    console.show_status('Launching session ...')
    self._session = tf.Session(graph=self._graph)
    console.show_status('Session launched')
    # Prepare some tools
    if config.save_model: self._saver = tf.train.Saver()
    if config.summary or config.hp_tuning:
      self._summary_writer = tf.summary.FileWriter(self.log_dir)

    # Try to load exist model
    load_flag, self._counter = self._load()
    if not load_flag:
      assert self._counter == 0
      # If checkpoint does not exist, initialize all variables
      self._session.run(tf.global_variables_initializer())
      # Add graph
      if config.summary: self._summary_writer.add_graph(self._session.graph)
      # Write model description to file
      if config.snapshot:
        description_path = os.path.join(self.snapshot_dir, 'description.txt')
        write_file(description_path, self.description)

    return load_flag

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

    feed_dict.update(self._get_status_feed_dict(is_training))

    return feed_dict

  def _get_status_feed_dict(self, is_training):
    is_training_tensor = self._is_training
    assert isinstance(is_training_tensor, tf.Tensor)
    feed_dict = {is_training_tensor: is_training}

    return feed_dict

  def _inter_cut(self, content, start_time=None):
    # If run on the cloud, do not show progress bar
    if not FLAGS.progress_bar:
      console.show_status(content)
      return

    console.clear_line()
    console.show_status(content)
    console.print_progress(progress=self._training_set.progress,
                           start_time=start_time)

  def _load(self):
    return load_checkpoint(self.ckpt_dir, self._session, self._saver)

  def _save(self, step):
    model_path = os.path.join(self.ckpt_dir, '{}.model'.format(self.model_name))
    save_checkpoint(model_path, self._session, self._saver, step)

  # endregion : Private Methods

  """For some reason, do not remove this line"""


if __name__ == '__main__':
  console.show_status('__main__')





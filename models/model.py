from __future__ import absolute_import

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .. import FLAGS

from .. import config
from .. import console
from .. import pedia

from ..utils.local import check_path
from ..utils.local import clear_paths
from ..utils.local import load_checkpoint
from ..utils.local import save_checkpoint

from ..utils.tfdata import TFData


class Model(object):
  """
  Base class of all kinds of models
  """
  model_name = 'default'

  def __init__(self, mark=None):
    self.mark = FLAGS.mark if mark is None else mark

    self._training_set = None
    self._test_set = None
    self._metric = None
    self._merged_summary = None

    self._session = None
    self._summary_writer = None
    self._saver = None

    self._loss = None
    self._train_step = None

    self._counter = None

    self._snapshot_function = None

  # region : Properties

  @property
  def log_dir(self):
    return check_path(config.record_dir, config.log_folder_name, self.mark)

  @property
  def ckpt_dir(self):
    return check_path(config.record_dir, config.ckpt_folder_name, self.mark)

  @property
  def snapshot_dir(self):
    return check_path(config.record_dir, config.snapshot_folder_name,
                       self.mark)

  # endregion : Properties

  def build(self):
    raise  NotImplementedError('build method not implemented')

  def train(self, epoch=1, batch_size=128, training_set=None,
            test_set=None, print_cycle=0, snapshot_cycle=0,
            snapshot_function=None):
    # Check data
    if training_set is not None:
      self._training_set = training_set
    if test_set is not None:
      self._test_set = test_set
    if self._training_set is None:
      raise ValueError('Data for training not found')
    elif not isinstance(training_set, TFData):
      raise TypeError('Data for training must be an instance of TFData')

    if snapshot_function is not None:
      if not callable(snapshot_function):
        raise ValueError('snapshot_function must be callable')
      self._snapshot_function = snapshot_function

    # Get epoch and batch size
    epoch = FLAGS.epoch if FLAGS.epoch > 0 else epoch
    batch_size = FLAGS.batch_size if FLAGS.batch_size > 0 else batch_size
    assert isinstance(self._training_set, TFData)
    self._training_set.set_batch_size(batch_size)

    # Show configurations
    console.show_status('Configurations:')
    console.supplement('Training set feature shape: {}'.format(
      self._training_set.features.shape))
    console.supplement('epochs: {}'.format(epoch))
    console.supplement('batch size: {}'.format(batch_size))

    # Do some preparation
    if self._session is None:
      self.launch_model()
    if self._merged_summary is None:
      self._merged_summary = tf.summary.merge_all()

    # Begin iteration
    with self._session.as_default():
      for epc in range(epoch):
        console.section('Epoch {}'.format(epc + 1))
        # Record epoch start time
        start_time = time.time()
        while True:
          # Update model
          train_loss, end_epoch_flag = self._update_model()
          # Increase counter
          self._counter += 1
          # Print status
          if print_cycle > 0 and np.mod(self._counter, print_cycle) == 0:
            self._print_progress(epc, start_time, train_loss)
          # Snapshot
          if snapshot_cycle > 0 and np.mod(self._counter, snapshot_cycle) == 0:
            self._snapshot()
          # Check flag
          if end_epoch_flag:
            console.clear_line()
            console.show_status('End of epoch. Elapsed time is ' 
                                '{:.1f} secs'.format(time.time() - start_time))
            # Show metric
            if self._metric is not None and self._test_set is not None:
              assert isinstance(self._metric, tf.Tensor)
              metric = self._metric.eval(self._get_default_feed_dict(
                self._test_set, train=False))
              console.show_status('{} on test set is {:.4f}'.format(
                pedia.memo[pedia.metric_name], metric))

            break

        # End of epoch
        self._save(self._counter)

    # End training
    console.clear_line()
    self._summary_writer.flush()
    self._summary_writer.close()
    self._session.close()

  def _update_model(self):
    assert isinstance(self._training_set, TFData)
    batch, flag = self._training_set.next_batch(shuffle=FLAGS.shuffle)
    feed_dict = self._get_default_feed_dict(batch, train=True)

    summary, loss, _ = self._session.run(
      [self._merged_summary, self._loss, self._train_step],
      feed_dict = feed_dict)

    assert isinstance(self._summary_writer, tf.summary.FileWriter)
    self._summary_writer.add_summary(summary, self._counter)

    return loss, flag

  def _print_progress(self, epc, start_time, train_loss):
    test_status = ''
    if self._test_set is not None and not config.block_test:
      feed_dict = self._get_default_feed_dict(self._test_set, False)
      test_loss = self._loss.eval(feed_dict)
      test_status = ', Test loss = {:.3f}'.format(test_loss)

    total_epoch = self._counter / self._training_set.batches_per_epoch
    console.clear_line()
    console.show_status(
      'Epoch {} [{:.1f} Total] Train loss = {:.3f}{}'.format(
        epc + 1, total_epoch, train_loss, test_status))

    console.print_progress(progress=self._training_set.progress,
                           start_time=start_time)

  def _snapshot(self):
    if self._snapshot_function is None:
      return

    fig = self._snapshot_function(self)
    epcs = 1.0 * self._counter / self._training_set.batches_per_epoch
    plt.savefig("{}/train_{:.1f}_epcs.png".format(self.snapshot_dir, epcs))
    plt.close(fig)

  def _define_loss(self, loss):
    if not isinstance(loss, tf.Tensor):
      raise TypeError('loss must be a tensor')
    self._loss = loss

  def _define_train_step(self, optimizer=None, var_list=None):
    if self._loss is None:
      raise ValueError('loss has not been defined yet')
    with tf.name_scope('Optimizer'):
      if optimizer is None:
        optimizer = tf.train.AdamOptimizer(1e-4)

      self._train_step = optimizer.minimize(loss=self._loss, var_list=var_list)

  @staticmethod
  def _get_default_feed_dict(batch, train):
    feed_dict = {}
    for tensor in tf.get_collection(pedia.default_feed_dict):
      if 'input' in tensor.name.lower():
        feed_dict[tensor] = batch[pedia.features]
      elif 'target' in tensor.name:
        feed_dict[tensor] = batch[pedia.targets]
      elif pedia.keep_prob in tensor.name:
        feed_dict[tensor] = pedia.memo[tensor.name] if not train else 1.0
      elif pedia.is_training in tensor.name:
        feed_dict[tensor] = train

    return feed_dict

  def _load(self):
    return load_checkpoint(self.ckpt_dir, self._session, self._saver)

  def _save(self, step):
    model_path = os.path.join(self.ckpt_dir, '{}.model'.format(self.model_name))
    save_checkpoint(model_path, self._session, self._saver, step)

  def launch_model(self, overwrite=False):
    # Before launch session, do some cleaning work
    if overwrite:
      clear_paths([self.log_dir, self.ckpt_dir, self.snapshot_dir])

    self._session = tf.Session()
    self._saver = tf.train.Saver()
    self._summary_writer = tf.summary.FileWriter(self.log_dir,
                                                 self._session.graph)
    # Try to load exist model
    load_flag, self._counter = self._load()
    if not load_flag:
      assert self._counter == 0
      # If checkpoint does not exist, initialize all variables
      self._session.run(tf.global_variables_initializer())

    return load_flag




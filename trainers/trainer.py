from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

import tensorflow as tf
import tframe as tfr

from tframe import console
from tframe import TFData
from tframe.enums import InputTypes, SaveMode
from tframe.utils import imtool
from tframe.core import with_graph
from tframe.config import Config, Flag

from tframe.trainers.metric import Metric


class Trainer(object):
  """Base class of trainer for training tframe models"""
  def __init__(
      self,
      model,
      training_set=None,
      validation_set=None,
      snapshot=None,
      probe=None):
    # Set model for trainer
    if not isinstance(model, tfr.models.Model):
      raise TypeError('!! model must be an instance of tframe Model')
    self.model = model

    # Date set attributes
    self._training_set = None
    self._validation_set = None
    self.set_data(training_set, validation_set)

    # Set callable attributes
    self._snapshot_function = self._check_callable(snapshot, 'snapshot')
    self._probe = self._check_callable(probe, 'probe')

    # Initiate trainer hub
    self.th = TrainerHub(self)

  # region : Properties

  @property
  def training_set(self):
    if self._training_set is not None:
      assert isinstance(self._training_set, TFData)
    return self._training_set

  @property
  def validation_set(self):
    if self._validation_set is not None:
      assert isinstance(self._validation_set, TFData)
    return self._validation_set

  @property
  def session(self):
    session = self.model.session
    assert isinstance(session, tf.Session)
    return session

  @property
  def counter(self):
    return self.model.counter

  @counter.setter
  def counter(self, value):
    self.model.counter = value

  @property
  def total_rounds(self):
    # TODO: Batch size must be kept the same among different trials
    return self.counter / self.training_set.batches_per_epoch

  # endregion : Properties

  # region : Public Methods

  def set_data(self, training_set=None, validation_set=None):
    if training_set is not None:
      self._check_data(training_set, 'training set')
      self._training_set = training_set
    if validation_set is not None:
      self._check_data(validation_set, 'validation set')
      self._validation_set = validation_set

  # endregion : Public Methods

  # region : Train

  @with_graph
  def train(self, hub=None, **kwargs):
    # Set trainer hub
    self._init_trainer_hub(hub, **kwargs)
    # Do some check-up
    self._check_data(), self._sanity_check(), self.th.sanity_check()
    # TODO: Set batch size
    # Run model's pre-train method
    self.model.pretrain(**kwargs)
    # Show configurations
    self._show_configurations()
    # Check model.session
    self._check_model()
    # TODO: merged summary

    # Train with graph
    with self.session.as_default(): self._outer_loop()

    # :: After training
    self._end_training()

  # region : Before training

  def _init_trainer_hub(self, hub, **kwargs):
    if hub is not None:
      # If th is provided
      if not isinstance(hub, TrainerHub):
        raise TypeError('!! config must be an instance of TrainerHub')
      self.th = hub
      self.th.trainer = self
    else: self.th.set_up(**kwargs)

  def _sanity_check(self):
    """Should be overrode by subclasses"""
    pass

  def _show_configurations(self):
    # TODO: to be modified
    console.show_status('Configurations:')
    console.supplement('Training set feature shape: {}'.format(
      self._training_set.features.shape))
    console.supplement('epochs: {}'.format(self.th.epoch))
    console.supplement('batch size: {}'.format(self.th.batch_size))

  def _check_model(self):
    if self.model.session is None:
      self.model.launch_model(self.th.overwrite)

  # endregion : Before training

  # region : During training

  def _outer_loop(self):
    hub = self.th
    for rnd in range(hub.total_outer_loops):
      console.section('{} {}'.format(hub.round_name, rnd + 1))
      hub.tic()
      # Begin inner loop
      self._inner_loop(rnd)
      # End of round
      console.show_status('End of {}. Elapsed time is {:.1f} secs'.format(
        hub.round_name, hub.toc()))
      # Maybe give a report on metric
      if hub.validation_on: self.model.metric.end_round()
      # Maybe save model
      if hub.save_mode is SaveMode.NAIVE:
        self.model.agent.save_model()
        console.show_status('Model saved')
      # Early stop
      if hub.stop: break

  def _inner_loop(self, rnd):
    # Begin iteration
    for batch in self._gen_batches():
      # Increase iteration counter
      self.counter += 1
      # Update model
      loss_dict = self.model.update_model(data_batch=batch)
      # Print progress
      self._print_progress(rnd, loss_dict)
      # Validation
      self._validate_model()
      # Take snapshot
      self._snapshot()

  def _gen_batches(self):
    if self.model.input_type is InputTypes.BATCH:
      batches = self.training_set.gen_batches(
        self.th.batch_size, self.th.shuffle)
    elif self.model.input_type is InputTypes.RNN_BATCH:
      batches = self.training_set.gen_rnn_batches(
        self.th.batch_size, self.th.num_steps)
    else:
      raise TypeError('!! Unknown input type {}'.format(self.model.input_type))
    return batches

  # endregion : During training

  # region : After training

  def _end_training(self):
    # If this is a hp-tuning task, write record summary
    if self.th.hp_tuning:
      assert not self.th.summary
      self.model.metric.write_record_summary()

    # Flush summary
    if self.th.summary or self.th.hp_tuning:
      self.model.agent.summary_writer.flush()

  # endregion : After training

  # endregion : Train

  # region : Private Methods

  def _check_data(self, data_set=None, name='dataset'):
    if data_set is None:
      data_set = self._training_set
      name = 'training set'
    if data_set is None: raise ValueError('!! {} not found'.format(name))
    if not isinstance(data_set, TFData):
      raise TypeError('!! {} must be an instance of TFData'.format(name))

  @staticmethod
  def _check_callable(f, name):
    if f is not None and not callable(f):
      raise TypeError('!! {} must be callable'.format(name))
    return f

  def _inter_cut(self, content, prompt='>>', start_time=None):
    # Clear progress bar
    if self.th.progress_bar: console.clear_line()
    # Show content
    print('{} {}'.format(prompt, content))
    # Print progress bar
    if self.th.progress_bar:
      assert isinstance(self._training_set, TFData)
      console.print_progress(progress=self._training_set.progress,
                             start_time=start_time)

  @staticmethod
  def _dict_to_string(dict_):
    assert isinstance(dict_, dict)
    string_array = ['{} = {:.3f}'.format(k, v) for k, v in dict_.items()]
    return ', '.join(string_array)

  def _print_progress(self, rnd, loss_dict):
    if self.th.print_cycle == 0: return
    if np.mod(self.counter - 1, self.th.print_cycle) != 0: return

    loss_string = self._dict_to_string(loss_dict)
    content = '{} {} ({:.1f} Total) {}'.format(
      self.th.round_name, rnd + 1, self.total_rounds, loss_string),
    self._inter_cut(content, prompt='[Train]', start_time=self.th.start_time)

  def _validate_model(self):
    if not self.th.validation_on: return
    if np.mod(self.counter, self.th.validate_cycle) != 0: return

    # Get metric
    metric_dict = self.model.validate_model(self.validation_set)
    metric = metric_dict.values()[0]
    new_record = self.model.metric.take_down(metric)
    prompt = '[New Record]' if new_record else '[Validate]'
    self._inter_cut(self._dict_to_string(metric_dict), prompt=prompt)

    # If necessary, save model
    if (new_record and self.th.save_model
        and self.th.save_mode is SaveMode.ON_RECORD
        and self.total_rounds > self.th.warm_up_rounds):
      self.model.agent.save_model()
      self._inter_cut('Model saved')

  def _snapshot(self):
    if not self.th.snapshot: return
    if not self.th.snapshot_cycle > 0: return
    if np.mod(self.counter - 1, self.th.snapshot_cycle) != 0: return

    fig = self._snapshot_function(self.model)
    unit = 'epcs' if self.th.round_name == 'Epoch' else 'rnds'
    filename = 'train_{:.2f}_{}.png'.format(self.total_rounds, unit)
    self.model.agent.save_plot(fig, filename)
    self._inter_cut("Images saved to '{}'".format(filename), '[Snapshot]')

  # endregion : Private Methods


class TrainerHub(Config):
  """Trainer Hub manages configurations for Trainer and stores status during
     training"""

  # region : Class Attributes

  epoch = Flag.integer(1, 'Epoch number to train')
  batch_size = Flag.integer(1, 'Batch size')
  num_steps = Flag.integer(1, 'Number of time steps')
  shuffle = Flag.boolean(True, 'Whether to shuffle')

  print_cycle = Flag.integer(0, 'Print cycle')
  validate_cycle = Flag.integer(0, 'Validate cycle')
  snapshot_cycle = Flag.integer(0, 'Snapshot cycle')
  match_cycle = Flag.integer(0, 'Match cycle for RL')

  early_stop = Flag.boolean(False, 'Early stop option')
  save_mode = Flag.enum(SaveMode.NAIVE, SaveMode,
                        "Save mode, \in  ['naive', 'on_record']")
  warm_up_rounds = Flag.integer(5, 'If save mode is on_record, model will not'
                                   'be saved until warm-up finishes')
  idle_tol = Flag.integer(20, 'Tolerance of idle rounds when early stop is on')

  round_name = Flag.string('Epoch', 'Name of outer loop during training')
  round = Flag.integer(1, 'General concept of total outer loops, used'
                          ' when outer loop is not called epochs')

  # endregion : Class Attributes

  def __init__(self, trainer=None):
    self.trainer = trainer
    self.record_rnd = 0
    # metric log is a list of list
    self.metric_log = []

    self._start_time = None
    self._stop = False

  # region : Properties

  @property
  def total_outer_loops(self):
    """In most supervised learning tasks, each outer training loop is called
       an epoch. If epoch is specified in config, it will be returned as
       total outer loops. In other tasks such as reinforcement learning,
       an outer loop may be called an episode. In this case, set 'total_rounds'
        in config instead of epoch."""
    assert 1 in (self.epoch, self.round)
    return max(self.epoch, self.round)

  @property
  def validation_on(self):
    metric = self.trainer.model.metric
    assert isinstance(metric, Metric)
    if not metric.activated: return False
    val_data = self.trainer.validation_set
    return val_data is not None and self.validate_cycle > 0

  @property
  def start_time(self):
    return self._start_time

  @property
  def stop(self):
    value = self._stop and self.early_stop
    self._stop = False
    return value

  # endregion : Properties

  # region : Public Methods

  def set_up(self, **kwargs):
    for key, arg in kwargs.items():
      if hasattr(self, key): self.__setattr__(key, arg)
      else: raise ValueError('!! can not resolve key {}'.format(key))

  def sanity_check(self):
    assert isinstance(self.trainer, Trainer)

  def tic(self):
    self._start_time = time.time()

  def toc(self):
    assert self._start_time is not None
    return time.time() - self._start_time

  def report(self):
    pass

  # endregion : Public Methods


# Register trainer hub
TrainerHub.register()



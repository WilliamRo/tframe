from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from collections import OrderedDict

import tensorflow as tf
import tframe as tfr

from tframe import checker
from tframe import console
from tframe import context
from tframe.data.base_classes import TFRData
from tframe.data.dataset import DataSet
from tframe.data.perpetual_machine import PerpetualMachine
from tframe.data.sequences.seq_set import SequenceSet
from tframe.enums import InputTypes, SaveMode
from tframe.core import with_graph
from tframe.configs.config_base import Config, Flag
from tframe.utils.maths.stat_tools import Statistic

from tframe.trainers.metric import Metric


class Trainer(object):
  """Base class of trainer for training tframe models.

     Model save mechanism when save_mode is
       (1) SaveMode.NAIVE:
           Model will be saved only at the end of each round naively
       (2) SaveMode.ON_RECORD:
           Model will be saved only when a new metric record appears
           after model finishes its warm-up rounds
   """
  HubClass = None
  def __init__(
      self,
      model,
      training_set=None,
      validation_set=None,
      snapshot=None,
      probe=None,
      evaluate=None,
      terminator=None,
      test_set=None,
  ):
    # Set model for trainer
    if not isinstance(model, tfr.models.Model):
      raise TypeError('!! model must be an instance of tframe Model')
    self.model = model

    # Date set attributes
    self._training_set = None
    self._validation_set = None
    self._test_set = None
    self.set_data(training_set, validation_set, test_set)

    # Set callable attributes
    self._snapshot_function = checker.check_callable(snapshot)
    self._probe = checker.check_callable(probe)
    self._evaluate = checker.check_callable(evaluate)

    # Initiate trainer hub
    self.th = TrainerHub(self)

    # Private Attributes
    self._record_count = 0
    self._warm_up = True
    self.loss_history = Statistic(max_length=self.th.hist_buffer_len)
    self.val_metric = Statistic(max_length=2)
    self.train_metric = Statistic(max_length=2)

    self.HubClass = TrainerHub
    if terminator is not None: assert callable(terminator)
    self._terminator = terminator

    # TODO
    context.trainer = self

  # region : Properties

  @property
  def training_set(self):
    if self._training_set is not None:
      assert isinstance(self._training_set, TFRData)
    return self._training_set

  @property
  def validation_set(self):
    if self._validation_set is not None:
      assert isinstance(self._validation_set, TFRData)
    return self._validation_set

  @property
  def test_set(self):
    if self._test_set is not None:
      assert isinstance(self._test_set, TFRData)
    return self._test_set

  @property
  def is_online(self):
    return isinstance(self.training_set, PerpetualMachine)

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
  def metric(self):
    return self.model.metric

  @property
  def total_rounds(self):
    # TODO: Batch size must be kept the same among different trials
    if self.th.round_length is None: return None
    return self.counter / self.th.round_length

  @property
  def graph(self):
    return self.model.graph

  @property
  def _save_model_when_record_appears(self):
    return (self.th.save_model and self.th.save_mode is SaveMode.ON_RECORD
            and not self._warm_up and not (self.th.at_most_save_once_per_round
                                           and self._record_count > 1))
  @property
  def _save_model_at_round_end(self):
    return self.th.save_model and self.th.save_mode is SaveMode.NAIVE

  @property
  def _save_model_at_training_end(self):
    return self.th.save_model and self.th.save_model_at_the_end

  # endregion : Properties

  # region : Public Methods

  def set_data(self, training_set=None, validation_set=None, test_set=None):
    if training_set is not None:
      self._check_data(training_set, 'training set')
      self._training_set = training_set
    if validation_set is not None:
      self._check_data(validation_set, 'validation set')
      self._validation_set = validation_set
    if test_set is not None:
      self._check_data(test_set, 'test set')
      self._test_set = test_set

  # endregion : Public Methods

  # region : Train

  @with_graph
  def train(self, hub=None, **kwargs):
    # Set trainer hub
    self._init_trainer_hub(hub, **kwargs)
    # Run model's pre-train method
    self.model.pretrain(**kwargs)
    # Do some check-up
    self._check_data(), self._sanity_check(), self.th.sanity_check()
    # Check model.session
    self._check_model()
    # Show configurations
    self._show_configurations()
    # Maybe take down some notes
    self._take_notes_before_loops()

    # Train with graph
    with self.session.as_default():
      rounds = self._outer_loop()

    # :: After training
    self._end_training(rounds)
    self._handle_notes()

  # region : Before training

  def _init_trainer_hub(self, hub, **kwargs):
    if hub is not None:
      # If th is provided
      if not isinstance(hub, self.HubClass): raise TypeError(
        '!! config must be an instance of {}'.format(self.HubClass))
      self.th = hub
      self.th.trainer = self
    else: self.th.set_up(**kwargs)

    # Get round length
    num_steps = (self.th.num_steps
                 if self.model.input_type is InputTypes.RNN_BATCH else None)
    self.th.round_length = self.training_set.get_round_length(
      self.th.batch_size, num_steps)
    def set_cycle(attr_name, num_per_round):
      assert hasattr(self.th, attr_name)
      if self.th.round_length is not None:
        setattr(self.th, attr_name, self.th.round_length // num_per_round)

    # Set progress bar
    if self.th.progress_bar:
      self.th.progress_bar = self.th.round_length is not None

    # Check validation cycle
    if self.th.validation_per_round > 0 and self.validation_set is not None:
      set_cycle('validate_cycle', self.th.validation_per_round)

    # Check probe cycle
    if self.th.probe_per_round > 0 and self._probe is not None:
      set_cycle('probe_cycle', self.th.probe_per_round)

    # Check note cycle
    if self.th.note_per_round > 0:
      set_cycle('note_cycle', self.th.note_per_round)
    if self.th.note_cycle == 0 and self.th.export_tensors_upon_validation:
      self.th.note_cycle = self.th.validate_cycle

    # Other setting
    if not self.th.warm_up:
      self._warm_up = False

  def _sanity_check(self):
    """Should be overrode by subclasses"""
    pass

  def _show_configurations(self):
    console.show_status('Configurations:')
    self.model.agent.take_notes('Configurations:', date_time=False)
    for config in self.th.config_strings:
      console.supplement(config)
      self.model.agent.take_notes('.. {}'.format(config), date_time=False)

  def _take_notes_before_loops(self):
    if not self.th.export_note: return

  def _check_model(self):
    if not self.model.launched:
      self.model.launch_model(self.th.overwrite)

  # endregion : Before training

  # region : During training

  def _outer_loop(self):
    hub = self.th
    rnd = 0
    for _ in range(hub.total_outer_loops):
      rnd += 1
      if self.is_online: console.section('Iterations Begin')
      else: console.section('{} {}'.format(hub.round_name, rnd))
      hub.tic()

      # Do inner loop
      self._inner_loop(rnd)
      # End of round
      if hub.progress_bar:
        console.show_status('End of {}. Elapsed time is {:.1f} secs'.format(
          hub.round_name, hub.toc()))
      # Maybe give a report on metric
      if not self.is_online and hub.validation_on:
        self.model.end_round(rnd)
        if self.metric.get_idle_rounds(rnd) > self.th.patience:
          self.th.raise_stop_flag()

      # Advanced strategy
      # self._advanced_strategy(rnd)
      # Export monitor info
      # if tfr.monitor.activated: tfr.monitor.export()

      # Maybe save model
      if self._save_model_at_round_end: self._save_model()
      # Early stop
      if hub.stop and self.model.bust(rnd): break
      # Force terminate
      if hub.force_terminate: break

    if hub.gather_note:
      if self.is_online:
        self.model.agent.put_down_criterion('Total Iterations', self.counter)
      else:
        self.model.agent.put_down_criterion('Total Rounds', rnd)
      # Evaluate the best model if necessary
      ds_dict = OrderedDict()
      if hub.evaluate_train_set: ds_dict['Train'] = self.training_set
      if hub.evaluate_val_set: ds_dict['Val'] = self.validation_set
      if hub.evaluate_test_set: ds_dict['Test'] = self.test_set
      if len(ds_dict) > 0:
        # Load the best model
        if hub.save_model:
          flag, _ = self.model.agent.load()
          assert flag
        # Evaluate the specified data sets
        for name, data_set in ds_dict.items():
          if not isinstance(data_set, TFRData):
            raise TypeError('!! {} is not a TFRData'.format(name))
          # TODO
          value = self.model.evaluate_model(
            data_set, batch_size=hub.val_batch_size)
          title = '{} {}'.format(name, self.metric.name)
          self.model.agent.put_down_criterion(title, value)
          self.model.agent.take_notes('{}: {:.2f}'.format(title, value))

    if self._save_model_at_training_end: self._save_model()

    return rnd

  def _inner_loop(self, rnd):
    self._record_count = 0
    # Begin iteration
    self.th.cursor = 0
    for i, batch in enumerate(self._gen_batches()):
      # Sanity check (make sure sequence batch is equal-length)
      self._check_data_batch(batch)
      # Increase iteration counter
      self.th.cursor += 1
      self.counter += 1
      # Update model
      loss_dict = self._update_model(batch)
      # Print progress
      self._print_progress(rnd, loss_dict)
      # Validation
      if self._validate_model(rnd) and self._save_model_when_record_appears:
        self._save_model(inter_cut=True)
      # Probe
      self._run_probe()
      # Take notes
      self._take_notes_for_export()

      # Take snapshot TODO: merge snapshot to probe
      # self._snapshot()

      # Check early stop condition
      if self.is_online:
        if self.th.max_iterations is not None:
          if i + 1 >= self.th.max_iterations:
            self.th.force_terminate = True
        if self.th.early_stop:
          if self.metric.get_idle_counts(self.counter) > self.th.patience:
            self.th.force_terminate = True
      # After probing, training process may be terminated
      if self.th.force_terminate: break
    if self._warm_up and self._record_count < self.th.warm_up_thres:
      self._warm_up = False

  # endregion : During training

  # region : After training

  def _end_training(self, rounds):
    if self.th.progress_bar: console.clear_line()
    # If this is a hp-tuning task, write record summary
    if self.th.hp_tuning:
      assert not self.th.summary
      self.metric.write_record_summary()
    # Flush summary
    if self.th.summary or self.th.hp_tuning:
      self.model.agent.summary_writer.flush()
    # Take notes
    if self.is_online:
      self.model.agent.take_notes(
        'End training after {} iterations'.format(self.counter))
    else:
      total_round = ('' if self.total_rounds is None
                     else ' ({:.1f} total)'.format(self.total_rounds))
      self.model.agent.take_notes(
        'End training after {} rounds{}'.format(rounds, total_round))
    # Evaluate
    if self._evaluate is not None: self._evaluate(self)

  def _handle_notes(self):
    # Add metric info into notes
    if self.th.validation_on: self.model.take_down_metric(self.is_online)
    # Put down key configurations to note
    self.model.agent.put_down_configs(self.th)

    # Show notes
    self.model.agent.show_notes()
    # Export notes if necessary
    if self.th.export_note:
      self.model.agent.export_notes()
    # Gather notes if necessary
    if self.th.gather_note:
      self.model.agent.gather_notes()

  # endregion : After training

  # endregion : Train

  # region : Private Methods

  def _update_model(self, data_batch):
    loss_dict = self.model.update_model(data_batch=data_batch)
    loss_slots = [s for s in loss_dict.keys() if s.name == 'Loss']
    assert len(loss_slots) > 0
    self.loss_history.record(loss_dict[loss_slots[0]])
    return loss_dict

  def _check_data(self, data_set=None, name='dataset'):
    if data_set is None:
      data_set = self._training_set
      name = 'training set'
    if data_set is None: raise ValueError('!! {} not found'.format(name))
    if not isinstance(data_set, TFRData):
      raise TypeError('!! {} must be an instance of TFRData'.format(name))

  @staticmethod
  def _check_callable(f, name=None):
    if name is None: return
    if f is not None and not callable(f):
      raise TypeError('!! {} must be callable'.format(name))
    return f

  def _gen_batches(self):
    """This method will be called only in the inner loop of train process."""
    if isinstance(self.training_set, SequenceSet):
      # TODO: for now a batch consists of sequences with different lengths can
      #  not be used for training for the padded 0s may produce inappropriate
      #  gradients.
      # if (self.th.batch_size > 1 and not self.training_set.parallel_on and
      #     self.training_set.batch_preprocessor is None):
      #   # a batch of equal-length sequences is allowed
      #   raise AssertionError('!! parallel engine is not activated')
      pass
    return self.model.get_data_batches(
      self.training_set, self.th.batch_size, self.th.num_steps,
      self.th.shuffle, is_training=True)

  @staticmethod
  def _check_data_batch(batch):
    assert isinstance(batch, DataSet)
    if batch.is_rnn_input and batch.active_length is not None:
      if max(batch.active_length) > min(batch.active_length):
        raise ValueError('!! Sequence batches must be equal-length')

  def _advanced_strategy(self, rnd):
    """Should be overridden"""
    pass

  def _inter_cut(self, content, prompt='>>', start_time=None):
    # Show content
    console.show_status(content, symbol=prompt)
    # Print progress bar
    if self.th.progress_bar and self.th.round_length is not None:
      assert isinstance(self._training_set, TFRData)
      progress = self.th.round_progress
      assert progress is not None
      console.print_progress(progress=progress, start_time=start_time)

  @staticmethod
  def _dict_to_string(dict_):
    assert isinstance(dict_, dict)
    string_array = ['{} = {:.3f}'.format(k, v) for k, v in dict_.items()]
    return ', '.join(string_array)

  def _print_progress(self, rnd, loss_dict):
    if loss_dict is None or self.th.print_cycle == 0: return
    if np.mod(self.counter - 1, self.th.print_cycle) != 0: return

    loss_string = self._dict_to_string(loss_dict)
    total_rounds = (' - ' if self.total_rounds is None else
                    ' ({:.1f} Total) '.format(self.total_rounds))
    if not self.is_online:
      content = '{} {}{}{}'.format(
        self.th.round_name, rnd, total_rounds, loss_string)
    else:
      content = 'Iteration {} - {}'.format(self.counter, loss_string)
    self._inter_cut(content, prompt='[Train]', start_time=self.th.start_time)

  def _get_tensors_to_export(self):
    """For now only RNN dynamics are tracked"""
    from tframe.models.recurrent import Recurrent
    from tframe.models.feedforward import Feedforward

    # This method is based on validation set
    if not self.th.validation_on: return OrderedDict()

    if self.model.input_type is InputTypes.RNN_BATCH:
      return Recurrent.get_tensor_to_export(self)
    else: return Feedforward.get_tensor_to_export(self)

  def _take_notes_for_export(self):
    if self.th.note_cycle == 0: return
    if np.mod(self.counter, self.th.note_cycle) != 0: return
    if not self.loss_history.last_value: return
    if self.th.validation_on:
      if not self.val_metric.last_value: return
      if self.th.validate_train_set and not self.train_metric.last_value: return

    # - Scalars
    scalars = OrderedDict()
    scalars['Loss'] = self.loss_history.running_average
    if self.train_metric.last_value:
      scalars['Train {}'.format(
        context.metric_name)] = self.train_metric.last_value
    if self.val_metric.last_value:
      scalars['Val {}'.format(
        context.metric_name)] = self.val_metric.last_value

    # - Tensors
    tensors = self._get_tensors_to_export()

    # Take down
    self.model.agent.take_down_scalars_and_tensors(
      scalars, tensors=tensors)
    self._inter_cut('Notes taken down.', prompt='[Export]')
    # For quickly note taking
    if self.th.terminate_on_note: self.th.force_terminate = True

  def _run_probe(self):
    if self._probe is None or self.th.probe_cycle == 0: return False
    if np.mod(self.counter, self.th.probe_cycle) != 0: return False
    # content = self._probe(self, loss_dict=loss_dict)
    content = self._probe(self)
    if content is None or content == '': return
    self._inter_cut(content, prompt='[Probe]', start_time=self.th.start_time)

  def _validate_model(self, rnd):
    if not self.th.validation_on: return False
    if np.mod(self.counter, self.th.validate_cycle) != 0: return False

    # Get metric
    metric_dict = self.model.validate_model(
      self.validation_set, self.th.val_batch_size, allow_sum=self.th.summary)

    new_record = None
    content_dict = OrderedDict()
    attachments = []
    if self.th.validate_train_set:
      train_dict = self.model.validate_model(
        self.training_set, self.th.val_batch_size, allow_sum=False)
      assert len(train_dict) == 1
      for slot, val in train_dict.items():
        assert isinstance(slot, Metric)
        key = 'Train {}'.format(slot.name[:3])
        content_dict[key] = val
        self.train_metric.record(val)
    # TODO: The code block below should be refactored
    for metric_slot, val in metric_dict.items():
      assert isinstance(metric_slot, Metric)
      self.val_metric.record(val)
      if new_record is None:
        new_record = self.metric.take_down(
          val, rnd, self.counter, gap=self.th.record_gap)
        # Terminator will check `val` if new_record appears
        if callable(self._terminator) and self._terminator(val):
          self.th.force_terminate = True

        key = ('Val {}'.format(metric_slot.name[:3])
               if self.th.validate_train_set else metric_slot.name)
        content_dict[key] = val
        # content = self._dict_to_string({metric_slot: val})
      else:
        # TODO: what's this for ??
        metric_slot.take_down(val, rnd, self.counter, gap=self.th.record_gap)
        attachments.append('{:.3f}'.format(val))
      if self.th.keep_trainer_log: self.th.logs[metric_slot.name] = val

    content = self._dict_to_string(content_dict)
    if len(attachments) > 0:
      content = '{} ({})'.format(content, ', '.join(attachments))
    if new_record:
      content += ' <New Record>'
      self._record_count += 1
    else:
      content += ' (Best: {:.3f})'.format(self.metric.record)
      if self.th.early_stop:
        idle = (self.metric.get_idle_counts(self.counter) if self.is_online
                else self.metric.get_idle_rounds(rnd))
        content = content[:-1] + ', Patience {}/{})'.format(
          idle, self.th.patience)
    self._inter_cut(content, prompt='[Validate]')

    return new_record

  def _snapshot(self):
    if not self.th.snapshot: return
    if not self.th.snapshot_cycle > 0: return
    if np.mod(self.counter - 1, self.th.snapshot_cycle) != 0: return

    fig = self._snapshot_function(self.model)
    step = self.counter if self.total_rounds is None else self.total_rounds
    filename = 'train_{:.2f}.png'.format(step)
    self.model.agent.save_plot(fig, filename)
    self._inter_cut("Images saved to '{}'".format(filename), '[Snapshot]')

  def _save_model(self, inter_cut=False):
    self.model.agent.save_model()
    print_method = self._inter_cut if inter_cut else console.show_status
    print_method('Model saved')

  # endregion : Private Methods


class TrainerHub(Config):
  """Trainer Hub manages configurations for Trainer and stores status during
     training"""

  # region : Class Attributes

  epoch = Flag.integer(1, 'Epoch number to train', is_key=None)
  max_iterations = Flag.integer(None, 'Max inner iterations')
  batch_size = Flag.integer(1, 'Batch size', is_key=None)
  num_steps = Flag.integer(None, 'Number of time steps', is_key=None)
  shuffle = Flag.boolean(False, 'Whether to shuffle', is_key=None)

  print_cycle = Flag.integer(0, 'Print cycle')
  validate_cycle = Flag.integer(0, 'Validate cycle')
  validation_per_round = Flag.integer(0, 'Validation per round',
                                      name='val_per_rnd')
  snapshot_cycle = Flag.integer(0, 'Snapshot cycle')
  probe_cycle = Flag.integer(0, 'Probe cycle')
  probe_per_round = Flag.integer(0, 'Probe per round')
  match_cycle = Flag.integer(0, 'Match cycle for RL')

  early_stop = Flag.boolean(False, 'Early stop option', is_key=None)
  record_gap = Flag.float(0.0, 'Minimum improvement')
  patience = Flag.integer(
    20, 'Tolerance of idle rounds(or iterations) when early stop is on',
    is_key=None)
  save_mode = Flag.enum(SaveMode.NAIVE, SaveMode,
                        "Save mode, \in  ['naive', 'on_record']")
  warm_up_thres = Flag.integer(1, 'Warm up threshold', is_key=None)
  warm_up = Flag.boolean(False, 'Whether to warm up')
  at_most_save_once_per_round = Flag.integer(False, '...')

  round_name = Flag.string('Epoch', 'Name of outer loop during training')
  round = Flag.integer(1, 'General concept of total outer loops, used'
                          ' when outer loop is not called epochs', is_key=None)
  hist_buffer_len = Flag.integer(
    20, 'Max length of historical statistics buffer length')
  validate_train_set = Flag.boolean(False, 'Whether to validate train set')
  terminal_threshold = Flag.float(0., 'Terminal threshold')

  # endregion : Class Attributes
  trainer_class = Trainer

  def __init__(self, trainer=None, as_global=False):
    # Call parent's constructor
    Config.__init__(self, as_global)

    self.trainer = trainer
    self.record_rnd = 0
    # metric log is a list of list
    self.metric_log = []

    self._start_time = None
    self._stop = False

    self.round_length = None
    self.cursor = None

    self.force_terminate = False
    # Sometimes probe method should know the accuracy history
    self.logs = {}

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
    metric = self.trainer.metric
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

  @property
  def round_progress(self):
    if self.round_length is None or self.cursor is None: return None
    return 1.0 * self.cursor / self.round_length

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

  def raise_stop_flag(self):
    self._stop = True

  # endregion : Public Methods

TrainerHub.register()

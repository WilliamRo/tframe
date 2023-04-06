from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from collections import OrderedDict

from tframe import tf
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
from tframe.core import Nomear
from tframe.configs.config_base import Config, Flag
from tframe.utils.maths.stat_tools import Statistic

from tframe.trainers.metrics_manager import MetricsManager


class Trainer(Nomear):
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
    self.model.metrics_manager.trainer = self

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
    self.batch_loss_stat = Statistic(max_length=self.th.hist_buffer_len)

    self.HubClass = TrainerHub
    if terminator is not None: assert callable(terminator)
    self._terminator = terminator

    # Important, since th.lives initialized by shell command will not change
    self._lives = self.th.lives

    # TODO
    # temporary solution to give agent the access to trainer
    context.trainer = self

  # region : Properties

  @property
  def key_metric(self):
    return self.metrics_manager.early_stop_slot

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
  def metrics_manager(self):
    assert isinstance(self.model.metrics_manager, MetricsManager)
    return self.model.metrics_manager

  @property
  def total_rounds(self):  # TODO: CC
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

  @property
  def recommend_decay_steps(self):
    bs = self.effective_batch_size
    epochs = self.th.patience if self.th.early_stop else self.th.epoch
    return self.training_set.size // bs * (epochs + 1)

  @property
  def effective_batch_size(self):
    if self.th.bs_mar in (None, 1.0): return self.th.batch_size
    assert self.th.bs_mar > 0
    return self.get_from_pocket(
      'EFFECTIVE_BATCH_SIZE', initializer=lambda: self.th.batch_size)

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

  def recover_progress(self, start_time=None):
    # Print progress bar
    if self.th.progress_bar and self.th.round_length is not None:
      assert isinstance(self._training_set, TFRData)
      progress = self.th.round_progress
      assert progress is not None
      console.print_progress(progress=progress, start_time=start_time)

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
      if self.th.save_model_in_the_beginning: self._save_model()
      rounds = self._outer_loop()

    # :: After training
    self._end_training(rounds)

    # Prune and save if necessary
    if self.th.prune_on: context.pruner.prune_and_save_lottery18()

    # Notes should be exported at the end
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

    # Set progress bar
    if self.th.progress_bar: self.th.progress_bar = self.th.round_len_is_active

    # Other setting
    if not self.th.warm_up: self._warm_up = False

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
    # Check model.epoch
    if not self.is_online and self.model.rounds is None: self.model.rounds = 0

  # endregion : Before training

  # region : During training

  def _outer_loop(self):
    hub = self.th
    rnd = 0

    # Record weights before training
    if hub.monitor_weight_history: context.monitor.record_weights()

    # Reset metrics if required
    if self.th.clear_records_before_training:
      self.model.metrics_manager.reset_records()

    # Validate model at the beginning if required
    if self.th.validate_at_the_beginning:
      self._validate_model(rnd=1)
      self._take_notes_for_export()
    # Set lr decay variables
    self._reset_lr_decay_variables()

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
      # Inc rounds for models training in epochs
      if self.model.rounds is not None:
        self.model.rounds += 1.0
      # Maybe give a report on metric
      if not self.is_online and hub.validation_on:
        self.model.end_round(rnd)
        if self.key_metric.get_idle_rounds(rnd) >= self.th.patience:
          self.th.raise_stop_flag()

      # Maybe save model (model.rounds var has been increased)
      if self._save_model_at_round_end: self._save_model()

      break_flag = False
      # Early stop via stop flag TODO: needed to be unified
      if hub.stop and self.model.bust(rnd): break_flag = True
      # Force terminate
      if hub.force_terminate: break_flag = True
      # Resurrect if possible
      if break_flag and self._lives > 0:
        self.resurrect(rnd)
        if not self.metrics_manager.resurrected:
          self.metrics_manager.resurrected = True
          self.metrics_manager.rar0 = self.metrics_manager.early_stop_criterion
        hub.force_terminate = False
        break_flag = False
      # Break if needed to
      if break_flag: break

    # Out of loop
    if hub.gather_note:
      if self.is_online:
        self.model.agent.put_down_criterion('Total Iterations', self.counter)
      else:
        self.model.agent.put_down_criterion('Total Rounds', rnd)

    # Put down final weight fraction if etch is on
    if self.th.etch_on:
      frac = context.pruner.weights_fraction
      self.model.agent.take_notes('Final weight fraction: {:.2f}%'.format(frac))
      self.model.agent.put_down_criterion('Weight Fraction', frac)

    # Evaluate the best model if necessary
    ds_dict = {ds.name: ds for ds in hub.datasets_for_evaluation}
    if hub.evaluate_train_set: ds_dict['Train'] = self.training_set
    if hub.evaluate_val_set: ds_dict['Val'] = self.validation_set
    if hub.evaluate_test_set: ds_dict['Test'] = self.test_set
    if len(ds_dict) > 0:
      # Load the best model
      if hub.save_model:
        flag, _, _ = self.model.agent.load()
        assert flag
      # Evaluate the specified data sets
      for name, data_set in ds_dict.items():
        if not isinstance(data_set, TFRData):
          raise TypeError('!! {} set is not a TFRData'.format(name))
        # TODO
        value = self.model.evaluate_model(
          data_set, batch_size=hub.eval_batch_size)
        title = '{} {}'.format(name, self.metrics_manager.eval_slot.name)
        self.model.agent.put_down_criterion(title, value)
        self.model.agent.take_notes('{}: {}'.format(title, hub.decimal_str(
          value, hub.val_decimals)))

    # Save model here if necessary
    if self._save_model_at_training_end:
      assert len(ds_dict) == 0
      self._save_model()

    # Update puller's omega if necessary
    if hub.cl_reg_on:
      context.puller.spring.update_omega_after_training()

    # Save shadow if required
    if hub.create_shadow_vars and hub.save_shadow_vars:
      self.model.agent.load()
      self.model.synchronize_shadow()
      self._save_model()

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
      # Increase lr global step if necessary
      if self.th.lr_decay_enabled: context.increase_lr_global_step()
      # Print progress
      self._print_progress(rnd, loss_dict)

      # Validation
      if self._validate_model(rnd) and self._save_model_when_record_appears:
        if not self.is_online: assert np.isscalar(self.th.round_progress)
        self._save_model(inter_cut=True, progress=self.th.round_progress)
      # Etch (i begins from 0, while rnd begins from 1)
      if self.is_online:
        if i >= self.th.etch_warm_up_steps: self._etch()
      elif rnd > self.th.etch_warm_up_rounds: self._etch()
      # Probe
      self._run_probe()
      # Take notes
      self._take_notes_for_export()

      # Check early stop condition
      if self.is_online:
        if self.th.max_iterations is not None:
          if i + 1 >= self.th.max_iterations:
            self.th.force_terminate = True
        if self.th.early_stop:
          if self.key_metric.get_idle_counts(self.counter) > self.th.patience:
            self.th.force_terminate = True
      # After probing, training process may be terminated
      if self.th.force_terminate:
        # If model will be resurrected later, dynamic_round_len if train_set
        # should be set to None. Otherwise error may occur TODO
        if hasattr(self.training_set, '_clear_dynamic_round_len'):
          # Perpetual Machine does not have this method
          self.training_set._clear_dynamic_round_len()
        break
    # Check warm up logic
    if self._warm_up and self._record_count < self.th.warm_up_thres:
      self._warm_up = False

  def _reset_lr_decay_variables(self):
    if not self.th.lr_decay_enabled: return
    context.reset_lr_global_step()
    context.set_lr_decay_steps(self.recommend_decay_steps)

  def resurrect(self, rnd):
    # Decrease lives by 1 and show status
    assert self._lives > 0
    self._lives -= 1
    console.show_status(
      'Lives decreased to {}'.format(self._lives), '[Resurrect]')
    console.show_status('Resurrecting ...')
    # [Compromise] set record counter or round
    self.key_metric.set_record_counter(self.counter)
    self.key_metric.set_record_round(rnd)

    # Load model
    flag, _, _ = self.model.agent.load()
    assert flag
    # Decay learning rate if necessary
    if self.th.lr_decay < 1.0:
      assert self.th.lr_decay > 0
      self.th.opt_lr_multiplier *= self.th.lr_decay
      if self.th.reset_optimizer_after_resurrection:
        self.model.reset_optimizer()

      self.model.set_train_step()
      console.show_status('Learning rate decayed to {:.6f}'.format(
        self.th.learning_rate * self.th.opt_lr_multiplier))

    # Modify batch size if required.
    # (bs_mar is short for `batch size modifier after resurrection`)
    if self.th.bs_mar not in (None, 1.0):
      assert self.th.bs_mar > 0
      self.replace_stuff(
        'EFFECTIVE_BATCH_SIZE', int(self.th.bs_mar * self.effective_batch_size))
      console.show_status('Batch size adjusted to {}'.format(
        self.effective_batch_size))

    # Reset global_step and decay_steps if necessary
    self._reset_lr_decay_variables()

  # endregion : During training

  # region : After training

  def _end_training(self, rounds):
    if self.th.progress_bar: console.clear_line()
    # If this is a hp-tuning task, write record summary
    if self.th.hp_tuning:
      assert not self.th.summary
      self.key_metric.write_record_summary()
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
    if self._evaluate is not None:
      # Load the best model if necessary
      if self.th.save_model:
        flag, _, _ = self.model.agent.load()
        assert flag
      # Evaluate model
      self._evaluate(self)
    # Show RAS if necessary
    if self.th.lives > 0:
      ras_info = self.metrics_manager.RAR_string
      console.show_status(ras_info)
      self.model.agent.take_notes(ras_info)

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
    if self.th.tic_toc: self.th.tic(key='__update')

    # TODO: currently, the dynamic ground-truth logic is implemented by
    #       a naive run-twice logic
    if self.th.use_dynamic_ground_truth:
      data_batch = self.th.dynamic_ground_truth_generator(
        self.model, data_batch)

    # Update model
    loss_dict = self.model.update_model(data_batch=data_batch)

    # Get and process loss slots
    # assert len(loss_slots) > 0
    loss_slots = [s for s in loss_dict.keys() if s.name == 'Loss']
    assert len(loss_slots) == 1
    loss_slot = loss_slots[0]
    self.batch_loss_stat.record(loss_dict[loss_slot])

    # Record grads if necessary
    # <monitor_grad_step_03: fetch and record>
    if self.th.monitor_weight_grads:
      grads = loss_dict.pop(self.model.grads_slot)
      context.monitor.record_grads(grads)

    # Monitor weight history for etching and cl-reg
    if self.th.monitor_weight_history:
      context.monitor.record_weights()

    # Do something for cl-reg if necessary
    if self.th.cl_reg_on:
      context.puller.spring.call_after_each_update()

    # Record other tensors
    if self.model.general_tensor_slot.activated:
      tensors = loss_dict.pop(self.model.general_tensor_slot)
      context.monitor.record_tensors(tensors)

    # (Temporary)
    if 'callback_model_updated' in context.depot:
      context.depot['callback_model_updated'](self)

    # Check NaN
    if self.th.terminate_on_nan:
      for val in loss_dict.values():
        if np.isnan(val):
          msg = 'Forced termination triggered due to NAN in loss_dict'
          console.show_status(msg)
          self.model.agent.take_notes(msg)
          self.th.force_terminate = True
          break

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
      self.training_set, self.effective_batch_size, self.th.num_steps,
      self.th.shuffle, is_training=True)

  @staticmethod
  def _check_data_batch(batch):
    assert isinstance(batch, DataSet)
    # The constraint below is not necessary due to gather_indices mechanism
    # if batch.is_rnn_input and batch.active_length is not None:
    #   if max(batch.active_length) > min(batch.active_length):
    #     raise ValueError('!! Sequence batches must be equal-length')

  def _advanced_strategy(self, rnd):
    """Should be overridden"""
    pass

  def _inter_cut(self, content, prompt='>>', start_time=None):
    # Show content
    console.show_status(content, symbol=prompt)
    # Print progress bar
    self.recover_progress(start_time)

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

    # Show time elapsed for a single update if required
    if self.th.tic_toc:
      # Here the key for toc should be taken care of
      # TODO: note that print progress may take a lot of time
      content += ' ({:.1f}ms)'.format(self.th.toc('__update') * 1000)

    self._inter_cut(content, prompt='[Train]', start_time=self.th.start_time)

  def _get_tensors_to_export(self):
    """For now only RNN dynamics are tracked"""
    from tframe.models.recurrent import Recurrent
    from tframe.models.feedforward import Feedforward

    # This method is based on validation set
    if not self.th.validation_on: return OrderedDict()

    if self.model.input_type is InputTypes.RNN_BATCH:
      tensor_dict = Recurrent.get_tensor_to_export(self)
    else: tensor_dict = Feedforward.get_tensor_to_export(self)

    # Add variables to export
    self._get_variables_to_export(tensor_dict)

    return tensor_dict

  def _get_variables_to_export(self, tensor_dict):
    if tensor_dict is None: tensor_dict = OrderedDict()
    assert isinstance(tensor_dict, dict)

    base_on_exemplars = len(tensor_dict) > 0
    # Compromise to avoid widget conflict in tensor_viewer
    def _add_to_dict(key, value):
      if base_on_exemplars:
        for exemplar_dict in tensor_dict.values():
          exemplar_dict[key] = value
      else: tensor_dict[key] = value

    # Add variables to export
    v_fetches_dict = context.variables_to_export
    if len(v_fetches_dict) > 0:
      results = self.model.agent.session.run(list(v_fetches_dict.values()))
      for key, value in zip(v_fetches_dict.keys(), results):
        _add_to_dict(key, value)

    # :: Add numpy arrays that stored in monitor
    # Add grads stats if necessary
    if self.th.export_weight_grads:
      for key, value in context.monitor.grad_dict.items():
        _add_to_dict(key, value)

    # Add general stats
    if self.th.export_activations:
      for key, value in context.monitor.stats_dict.items():
        _add_to_dict(key, value)

    # Add customized tensors to export
    if callable(self.th.customized_tensors_to_export):
      for key, value in self.th.customized_tensors_to_export().items():
        _add_to_dict(key, value)

    return tensor_dict

  def _take_notes_for_export(self):
    # Note switch should be turned on
    if self.th.note_modulus == 0: return
    # Note cycle should be met
    # self.counter == 0 happens when th.validate_at_the_beginning is True
    if self.counter != 0:
      if np.mod(self.counter, self.th.note_modulus) != 0: return
      # if not (self.counter == 1 and self.th.take_note_in_beginning): return
    # Loss history should not be blank
    # if not self.batch_loss_stat.last_value: return  # TODO
    # Validation history should not be blank if validation is on
    if self.th.validation_on:
      if not self.metrics_manager.ready_for_note_taking: return

    # - Scalars
    scalars = OrderedDict()
    scalars['Loss'] = self.batch_loss_stat.running_average
    self.metrics_manager.update_scalar_dict(scalars)
    if self.th.etch_on:
      scalars['Weight Fraction'] = context.pruner.weights_fraction

    # - Tensors
    tensors = self._get_tensors_to_export()
    # Take down
    self.model.agent.take_down_scalars_and_tensors(
      scalars, tensors=tensors)
    self._inter_cut('Notes taken down.', prompt='[Export]')
    # For quickly note taking
    if self.th.terminate_on_note: self.th.force_terminate = True

  def _run_probe(self):
    if self._probe is None or self.th.probe_modulus == 0: return False
    if np.mod(self.counter, self.th.probe_modulus) != 0: return False
    # content = self._probe(self, loss_dict=loss_dict)
    content = self._probe(self)
    if content is None or content == '': return
    self._inter_cut(content, prompt='[Probe]', start_time=self.th.start_time)

  def _etch(self):
    if not self.th.etch_on: return
    if np.mod(self.counter, self.th.etch_modulus) != 0: return
    pruner = context.pruner
    assert pruner is not None
    pruner.etch_all()

  def _validate_model_on(self, dataset: TFRData):
    return self.model.validate_model(
      dataset, self.th.val_batch_size, allow_sum=self.th.summary,
      verbose=self.th.val_progress_bar, seq_detail=self.th.val_info_splits > 0)

  def _validate_model(self, rnd):
    if not self.th.validation_on: return False
    # Validate cycle should be met
    if self.counter == 0:
      if not (self.th.take_note_in_beginning
              or self.th.validate_at_the_beginning): return False
    elif np.mod(self.counter, self.th.validate_modulus) != 0: return False

    # Validate datasets other than validation set
    other_sets = []
    other_sets.extend(self.th.additional_datasets_for_validation)
    if self.th.validate_train_set: other_sets.append(self.training_set)
    if self.th.validate_test_set: other_sets.append(self.test_set)

    for ds in other_sets:
      res_dict = self._validate_model_on(ds)
      # Record
      self.metrics_manager.record_stats_on_dataset(ds, res_dict)

    # Validate val_set and record
    if self.th.tic_toc: self.th.tic('__validate')
    val_dict = self._validate_model_on(self.validation_set)

    if self.th.tic_toc:
      time_elapsed = self.th.toc('__validate') * 1000
      console.show_status(
        f'{time_elapsed:.1f}ms for {self.validation_set.size} samples',
        '[Tic-toc]')

    new_record = self.metrics_manager.record_stats_on_dataset(
      self.validation_set, val_dict, True, rnd)
    # Terminator will check early_stop_criterion if new_record appears
    if new_record and callable(self._terminator):
      if self._terminator(self.metrics_manager.early_stop_criterion):
        self.th.force_terminate = True
    # If lottery is on, take down criteria at the beginning
    if self.th.prune_on and self.counter == 1:
      for k, v in val_dict.items():
        self.model.agent.put_down_criterion(k.name + '-0', v)
    if self.th.etch_on:
      self.model.agent.put_down_criterion(
        'Weight Fraction', context.pruner.weights_fraction)

    # Print stats and return new_record flag
    self.metrics_manager.print_latest_stats(
      '[Validate]', decimals=self.th.val_decimals)
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

  def _save_model(self, inter_cut=False, progress=None):
    # Update model rounds
    total_rounds = None
    if not self.is_online:
      assert np.isscalar(self.model.rounds)
      total_rounds = self.model.rounds
      if progress is not None:
        assert 0 <= progress <= 1
        total_rounds += progress
    # Save model
    self.model.agent.save_model(rounds=total_rounds, suffix='train')
    # Show status
    print_method = self._inter_cut if inter_cut else console.show_status
    print_method('Model saved')

  # endregion : Private Methods

  # region : Public Methods

  def get_variables_to_export(self, export_dict=None):
    """This api is for customized probe method"""
    return self._get_variables_to_export(export_dict)

  # endregion : Public Methods


class TrainerHub(Config):
  """Trainer Hub manages configurations for Trainer and stores status during
     training"""

  # region : Class Attributes

  epoch = Flag.integer(1, 'Epoch number to train', is_key=None)
  max_iterations = Flag.integer(None, 'Max inner iterations')
  batch_size = Flag.integer(1, 'Batch size', is_key=None, hp_scale='log')
  num_steps = Flag.integer(None, 'Number of time steps', is_key=None)
  shuffle = Flag.boolean(True, 'Whether to shuffle', is_key=None)

  print_cycle = Flag.integer(0, 'Print cycle')
  validate_cycle = Flag.integer(0, 'Validate cycle')
  validate_at_the_beginning = Flag.boolean(
    False, 'Whether to validate before outer_loop')
  validation_per_round = Flag.integer(0, 'Validation per round',
                                      name='val_per_rnd')
  snapshot_cycle = Flag.integer(0, 'Snapshot cycle')
  probe_cycle = Flag.integer(0, 'Probe cycle')
  probe_per_round = Flag.integer(0, 'Probe per round')
  match_cycle = Flag.integer(0, 'Match cycle for RL')

  etch_per_round = Flag.integer(0, 'Etch per round')
  etch_cycle = Flag.integer(0, 'Etch cycle', is_key=None)

  early_stop = Flag.boolean(False, 'Early stop option', is_key=None)
  record_gap = Flag.float(0.0, 'Minimum improvement')
  patience = Flag.integer(
    20, 'Tolerance of idle rounds(or iterations) when early stop is on',
    is_key=None)
  save_mode = Flag.enum(SaveMode.ON_RECORD, SaveMode,
                        "Save mode, \in  ['naive', 'on_record']")
  warm_up_thres = Flag.integer(1, 'Warm up threshold', is_key=None)
  warm_up = Flag.boolean(False, 'Whether to warm up')
  at_most_save_once_per_round = Flag.integer(False, '...')

  round_name = Flag.string('Epoch', 'Name of outer loop during training')
  round = Flag.integer(1, 'General concept of total outer loops, used'
                          ' when outer loop is not called epochs', is_key=None)
  hist_buffer_len = Flag.integer(
    20, 'Max length of historical statistics buffer length')
  validate_train_set = Flag.boolean(
    False, 'Whether to validate train set in trainer._validate_model')
  validate_test_set = Flag.boolean(
    False, 'Whether to test train set in trainer._validate_model')
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

    self._time_stamp = {'__start_time': None}
    self._stop = False

    self._round_length = None
    self.cursor = None

    self.force_terminate = False
    # Sometimes probe method should know the accuracy history
    self.logs = {}

  # region : Properties

  @Config.property()
  def additional_datasets_for_validation(self): return []

  @property
  def round_length(self):
    assert isinstance(self.trainer.training_set, TFRData)
    # For being compatible with old versions
    if hasattr(self.trainer.training_set, 'dynamic_round_len'):
      return self.trainer.training_set.dynamic_round_len
    else: return getattr(self.trainer.training_set, '_dynamic_round_len', None)

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
    mm = self.trainer.metrics_manager
    assert isinstance(mm, MetricsManager)
    if not mm.has_metric: return False
    val_data = self.trainer.validation_set

    # return val_data is not None and self.validate_modulus > 0
    return all([val_data is not None,
                self.validate_cycle > 0 or self.validation_per_round > 0])

  @property
  def start_time(self):
    return self._time_stamp['__start_time']

  @property
  def stop(self):
    value = self._stop and self.early_stop
    self._stop = False
    return value

  @property
  def round_progress(self):
    if self.round_length is None or self.cursor is None: return None
    return 1.0 * self.cursor / self.round_length

  # region : Modulus

  @property
  def round_len_is_active(self):
    assert isinstance(self.trainer.training_set, TFRData)
    return not isinstance(self.trainer.training_set, PerpetualMachine)

  def _get_modulus(self, verb, act_per_round_key=None, act_cycle_key=None):
    assert isinstance(verb, str)
    if act_per_round_key is None:
      act_per_round_key = '{}_per_round'.format(verb)
    if act_cycle_key is None: act_cycle_key = '{}_cycle'.format(verb)
    # Get value
    act_per_round = getattr(self, act_per_round_key)
    act_cycle = getattr(self, act_cycle_key)
    # act_cycle has the highest priority
    if any([act_cycle > 0, not self.round_len_is_active, act_per_round <= 0]):
      return act_cycle
    # [Compromise] avoid error in Trainer._show_configuration method
    if self.round_length is None: return None
    return self.round_length // act_per_round

  @property
  def validate_modulus(self):
    return self._get_modulus(
      'validate', act_per_round_key='validation_per_round')

  @property
  def probe_modulus(self): return self._get_modulus('probe')

  @property
  def etch_modulus(self): return self._get_modulus('etch')

  @property
  def note_modulus(self):
    if self.note_cycle <= 0 and self.export_tensors_upon_validation:
      return self.validate_modulus
    return self._get_modulus('note')

  # endregion : Modulus

  # endregion : Properties

  # region : Public Methods

  def set_up(self, **kwargs):
    for key, arg in kwargs.items():
      if hasattr(self, key): self.__setattr__(key, arg)
      else: raise ValueError('!! can not resolve key {}'.format(key))

  def sanity_check(self):
    assert isinstance(self.trainer, Trainer)

  def tic(self, key='__start_time'):
    self._time_stamp[key] = time.time()

  def toc(self, key='__start_time'):
    assert self._time_stamp[key] is not None
    return time.time() - self._time_stamp[key]

  def raise_stop_flag(self):
    self._stop = True

  # endregion : Public Methods

TrainerHub.register()

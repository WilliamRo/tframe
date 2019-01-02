from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from collections import OrderedDict

import tframe as tfr
from tframe import DataSet
from tframe import hub
from tframe import checker
from tframe import console
from tframe import pedia

from tframe.enums import InputTypes
from tframe.core import with_graph
from tframe.core import TensorSlot, NestedTensorSlot
from tframe.core import SummarySlot, OperationSlot, IndependentSummarySlot
from tframe.core import Group
from tframe.core.agent import Agent

from tframe.trainers.metric import Metric
from tframe.trainers.scheme import TrainScheme
from tframe.trainers.trainer import Trainer, TrainerHub
from tframe.trainers.smartrainer import SmartTrainer, SmartTrainerHub

from tframe.data.base_classes import TFRData
from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.bigdata import BigData
from tframe.data.perpetual_machine import PerpetualMachine


class Model(object):
  """
  Base class of [all?] kinds of models built on TensorFlow
  """
  model_name = 'default'

  def __init__(self, mark=None):
    # Model mark usually helps to decide the folder name
    self.mark = hub.mark or mark
    assert mark is not None
    if hub.prefix is not None: self.mark = hub.prefix + self.mark
    if hub.suffix is not None: self.mark += hub.suffix
    hub.mark = self.mark

    # Each model has an agent to deal with some tensorflow stuff
    self.agent = Agent(self)

    # Define slots
    self._outputs = TensorSlot(self)

    self._metric = Metric(self, 'metric')
    self._validation_summary = SummarySlot(self)
    self._batch_val_summ = IndependentSummarySlot(self, 'batch_metric_summ')
    self._validate_group = Group(
      self, self._metric, self._validation_summary, name='Validate-group')

    self._loss = TensorSlot(self, 'Loss')
    self._train_step = OperationSlot(self)
    self._train_step_summary = SummarySlot(self)
    self._update_group = Group(
      self, self._loss, self._metric, self._train_step,
      self._train_step_summary, name='Update-group')

    # Private attributes
    self._default_net = None
    self._optimizer = None
    self._built = False
    self._scheme = None

    # Public attributes
    self.counter = None
    self.launched = False

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
  def outputs(self):
    assert isinstance(self._outputs, TensorSlot)
    return self._outputs

  @property
  def loss(self):
    assert isinstance(self._loss, TensorSlot)
    return self._loss

  @property
  def train_step(self):
    assert isinstance(self._train_step, OperationSlot)
    return self._train_step

  @property
  def built(self):
    assert isinstance(self._built, bool)
    return self._built

  @property
  def record(self):
    if not self.metric.activated: return None
    else: return self.metric.record

  @property
  def variable_to_save(self):
    """Should be called in with_graph decorator"""
    vars = (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) +
            tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))
    return [var for var in vars
            if var not in tf.get_collection(pedia.do_not_save)]

  @property
  def metric_foreach(self):
    metrics = tf.get_collection(pedia.metric_foreach)
    assert len(metrics) == 1
    return metrics[0]

  @property
  def parameters_dict(self):
    # Fetch all trainable variables
    trainable_variables = tf.trainable_variables()
    values = self.session.run(trainable_variables)
    # Wrap them into a dictionary and return
    parameters = {}
    for t, v, in zip(trainable_variables, values):
      parameters[t.name] = v
    return parameters

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

  @with_graph
  def build(self, optimizer=None, **kwargs):
    # Smooth out flags before important actions
    hub.smooth_out_conflicts()
    #
    self._build(optimizer=optimizer, **kwargs)
    # Initialize monitor
    self._init_monitor()
    # Set built flag
    self._built = True
    # Show build info
    console.show_status('Model built successfully:')
    description = self.description
    if not isinstance(description, (tuple, list)):
      description = [description]
    for line in description:
      assert isinstance(line, str)
      console.supplement(line)
    # Get structure detail TODO:
    detail, total_params = '', 0
    if hasattr(self, 'structure_detail'):
      detail, total_params = self.structure_detail
    # Maybe take some notes
    self.agent.take_notes('Model built successfully')
    self.agent.take_notes('Structure:', date_time=False)
    self.agent.take_notes('Total params: {}'.format(total_params))
    for line in description:
      self.agent.take_notes(line, date_time=False)
    if hub.show_structure_detail:
      print('.. Structure detail:\n{}'.format(detail))

  def _build(self, optimizer=None, **kwargs):
    """Abstract method, must be implemented in different models
       Usually touches tensorflow api directly and plug tf ops into tfr slots
    """
    raise  NotImplementedError('!! build method not implemented')

  def _init_monitor(self):
    if tfr.monitor.activated: tfr.monitor.init_monitor(self)

  @with_graph
  def _define_train_step(self, optimizer=None, var_list=None):
    """ TODO: should be modified for tframe.optimizer
        self._train_step will be plugged only here
    """
    if not self._loss.activated:
      raise AssertionError('!! loss has not been activated yet')
    with tf.name_scope('Optimizer'):
      if optimizer is None: optimizer = tf.train.AdamOptimizer(1e-4)

      # TODO: BETA
      if hub.use_rtrl:
        from tframe.optimizers.rtrl_opt import RealTimeOptimizer
        optimizer = RealTimeOptimizer(self, optimizer)

      self._optimizer = optimizer
      self._train_step.plug(
        optimizer.minimize(self._loss.op, var_list=var_list))

  def _merge_summaries(self):
    train_step_summaries = tf.get_collection(pedia.train_step_summaries)
    validation_summaries = tf.get_collection(pedia.validation_summaries)
    if len(train_step_summaries) > 0:
      self._train_step_summary.plug(tf.summary.merge(train_step_summaries))
    if len(validation_summaries) > 0:
      self._validation_summary.plug(tf.summary.merge(validation_summaries))

  # endregion : Building

  # region : Training

  def pretrain(self, **kwargs):
    """Method run in early training process, should be overrode"""
    if self._scheme is not None:
      assert isinstance(self._scheme, TrainScheme)
      trial = self._scheme.dequeue()
      if trial is not None: trial.initialize(self)

  @with_graph
  def train(self, training_set, trainer_hub=None, validation_set=None,
            snapshot=None, probe=None, **kwargs):
    if trainer_hub is None:
      trainer_class = SmartTrainer if hub.smart_train else Trainer
    else:
      if not isinstance(trainer_hub, TrainerHub):
        raise TypeError('!! Input hub must be an instance of TrainerHub')
      trainer_class = trainer_hub.trainer_class
    trainer = trainer_class(
      self, training_set=training_set, validation_set=validation_set,
      snapshot=snapshot, probe=probe)
    trainer.train(hub=trainer_hub, **kwargs)

  def update_model(self, data_batch, **kwargs):
    """Default model updating method, should be overrode"""
    feed_dict = self._get_default_feed_dict(data_batch, is_training=True)
    return self._update_group.run(feed_dict)

  def get_data_batches(self, data_set, batch_size, num_steps=None,
                       shuffle=False):
    """ Get batch generator.
    :param data_set: an instance of DataSet or BigData from which data batches
                      will be extracted
    :param batch_size: if is None, default value will be assigned according to
                        the input type of this model
    :param num_steps: step number for RNN data batches
    :param shuffle: whether to shuffle
    :return: a generator or a list
    """
    # Data set must be an instance of DataSet or BigData
    assert isinstance(data_set, (DataSet, BigData, PerpetualMachine))
    if self.input_type is InputTypes.BATCH:
      # If model's input type is normal batch, num_steps will be ignored
      # If batch size is not specified and data is a DataSet, feed it all at
      #  once into model
      if batch_size is None and isinstance(data_set, DataSet):
        return [data_set.stack]
      checker.check_positive_integer(batch_size)
      data_batches = data_set.gen_batches(batch_size, shuffle=shuffle)

    elif self.input_type is InputTypes.RNN_BATCH:
      if num_steps is None: num_steps = -1
      if batch_size is None: batch_size = 1
      if batch_size < 0: batch_size = data_set.size

      # if batch_size > 1: assert num_steps < 0
      # it's legal for common DataSet to have num_steps > 0 while batch_size > 1
      if batch_size > 1 and isinstance(data_set, SequenceSet):
        assert num_steps < 0

      checker.check_positive_integer(batch_size)
      checker.check_type(num_steps, int)
      data_batches = data_set.gen_rnn_batches(batch_size, num_steps, shuffle)
    else: raise ValueError('!! Can not resolve input type of this model')

    return data_batches

  def validate_model(self, data, batch_size=None, allow_sum=False):
    """Validate model. If data provided is not regular, batch validation will
       be used. For RNN model, batch validation requires batch size to be 1."""
    # If data items are not regular arrays, it will be forced to carry out
    # .. batch validation
    assert isinstance(data, TFRData)
    if not data.is_regular_array and batch_size is None: batch_size = 1

    # Normal validation
    if batch_size is None:
      data = self._sanity_check_before_use(data)
      feed_dict = self._get_default_feed_dict(data, is_training=False)
      return self._validate_group.run(feed_dict, allow_sum=allow_sum)

    # Batch validation: Calculate metric batch by batch
    metric_list = []
    total = 0
    for batch in self.get_data_batches(data, batch_size, -1, False):
      # Batch validation on irregular data
      if batch.active_length is not None:
        # TODO: need a corresponding mechanism to ensure metric_foreach
        #       exists
        results = self._get_active_tensor(batch, self.metric_foreach)
        for result in results:
          if not batch.n_to_one:
            assert isinstance(result, np.ndarray) and len(result.shape) == 1
            metric_list.append(sum(result))
            total += len(result)
          else:
            if isinstance(result, np.ndarray):
              assert len(result) == 1
              result = result[0]
            assert np.isscalar(result)
            metric_list.append(result)
            total += 1
        continue

      # Calculate weight
      weight = batch.targets.shape[0]
      if self.input_type is InputTypes.RNN_BATCH:
        # shape of RNN batch targets is (batch_size, num_steps, *target_dim)
        weight *= batch.targets.shape[1]
      assert weight > 0
      total += weight

      # Validate batch
      batch = self._sanity_check_before_use(batch)
      feed_dict = self._get_default_feed_dict(batch, is_training=False)
      metric_list.append(self._metric.run(feed_dict) * weight)

    # Return metric mean
    metric_mean = np.sum(metric_list) / total
    if allow_sum: self._batch_val_summ.write(metric_mean)
    return {self._metric: metric_mean}

  def take_down_metric(self):
    if not self.metric.activated: return
    notes = 'Record: {:.3f}, Mean Record: {:.3f}'.format(
      self.metric.record, self.metric.mean_record)
    self.agent.take_notes(notes, date_time=False)
    # Add history into notes if necessary
    if hub.show_record_history_in_note:
      self.agent.take_notes(
        self.metric.metric_mean_history_str, date_time=False)
    # Add record and mean record to notes
    self.agent.put_down_criterion('Record', self.metric.record)
    self.agent.put_down_criterion('Mean Record', self.metric.mean_record)

  def end_round(self, rnd):
    self.metric.end_round(rnd)

  def bust(self, rnd):
    if self._scheme is not None:
      assert isinstance(self._scheme, TrainScheme)
      trial = self._scheme.dequeue()
      if trial is not None:
        trial.initialize(self)
        return False
      else: return True
    return True

  # endregion : Training

  # region : Public Methods

  def get_trainable_variables(self, f=None):
    if f is None: f = lambda _: True
    variables = [v for v in tf.trainable_variables() if f(v)]
    values = self.session.run(variables)
    variable_dict = OrderedDict()
    for t, v in zip(variables, values):
      variable_dict[t.name] = v
    return variable_dict

  # def run_for_tensors(self, tensors, data_batch):
  #   # Sanity check
  #   checker.check_type(tensors, tf.Tensor) and isinstance(data_batch, TFRData)

  def tune_lr(self, new_lr=None, coef=1.0):
    #TODO
    if self._optimizer is None:
      raise ValueError('!! Optimizer not defined yet')
    if self._optimizer.__class__ in [tf.train.AdamOptimizer]:
      lr_name = '_lr'
    elif self._optimizer.__class__ in [tf.train.GradientDescentOptimizer]:
      lr_name = '_learning_rate'
    else:
      raise TypeError('!! Unsupported optimizer for lr tuning')

    old_lr = self._optimizer.__getattribute__(lr_name)
    if new_lr is None: new_lr = old_lr * coef
    self._optimizer.__setattr__(lr_name, new_lr)

    # Show status
    console.show_status(
      'Learning rate updated: {:.2e} => {:.2e}'.format(old_lr, new_lr))

    return new_lr

  def set_scheme(self, scheme):
    if not isinstance(scheme, TrainScheme):
      raise TypeError('!! scheme must be an instance of TrainScheme')
    self._scheme = scheme

  def shutdown(self):
    self.agent.shutdown()

  def launch_model(self, overwrite=False):
    return self.agent.launch_model(overwrite)

  def batch_evaluation(
      self, fetches, data_set, batch_size=None, extractor=None):
    # Sanity check
    checker.check_type(fetches, tf.Tensor)
    assert isinstance(data_set, TFRData)
    fetches_is_single = not isinstance(fetches, (tuple, list))
    # Wrap fetches into a list if necessary
    if fetches_is_single: fetches = [fetches]

    # Define outputs as list of arrays/list of arrays
    outputs = [[] for _ in fetches]
    for data_batch in self.get_data_batches(data_set, batch_size):
      # Calculate output
      data_batch = self._sanity_check_before_use(data_batch)
      # TODO: consider n_to_one cases
      batch_outputs = self._get_active_tensor(data_batch, fetches)
      assert isinstance(batch_outputs, (tuple, list))
      # Extract if necessary
      if extractor is not None:
        assert callable(extractor)
        batch_outputs = [extractor(bo) for bo in batch_outputs]
      # Add output to outputs accordingly
      for output, batch_output in zip(outputs, batch_outputs):
        assert isinstance(output, list)
        if self.input_type is InputTypes.RNN_BATCH: output += batch_output
        else: output.append(batch_output)

    # Concatenate if possible
    if self.input_type is InputTypes.BATCH:
      outputs = [np.concatenate(o) for o in outputs]
    # Return
    if fetches_is_single: outputs = outputs[0]
    return outputs

  # endregion : Public Methods

  # region : Private Methods

  def _get_active_tensor(self, batch, fetches):
    checker.check_type(fetches, tf.Tensor)
    assert isinstance(batch, DataSet)
    fetches_is_single = not isinstance(fetches, (tuple, list))

    feed_dict = self._get_default_feed_dict(batch, is_training=False)
    values = self.session.run(fetches, feed_dict)
    checker.check_type(values, np.ndarray)

    # If values is a single element, wrap it into a list for potential slicing
    if fetches_is_single: values = [values]

    al = batch.active_length
    if self.input_type is InputTypes.RNN_BATCH:
      if al is not None:
        assert isinstance(al, list) and len(al) == batch.size
        outputs = [[y[:l] for y, l in zip(value, al)] for value in values]
        # If this model is for classifying a whole sequence
        if batch.n_to_one:
          outputs = [[y[-1] for y in output] for output in outputs]
      else: outputs = [[value] for value in values]
    else: outputs = values

    if fetches_is_single: outputs = outputs[0]
    return outputs

  @with_graph
  def _get_default_feed_dict(self, batch, is_training):
    feed_dict = {}
    for tensor in tf.get_collection(pedia.default_feed_dict):
      if 'input' in tensor.name.lower():
        feed_dict[tensor] = batch[pedia.features]
      elif 'target' in tensor.name:
        # TODO: when predict without outputing loss ...
        if batch.targets is not None: feed_dict[tensor] = batch.targets
      else:
        name = tensor.name.split('/')[-1].split(':')[0]
        val = batch.data_dict.get(name, None)
        if val is not None: feed_dict[tensor] = val

    feed_dict.update(self.agent.get_status_feed_dict(is_training))

    return feed_dict

  def _sanity_check_before_use(self, data):
    if not isinstance(data, DataSet):
      raise TypeError('!! Input data must be an instance of DataSet')
    if not self.built: raise ValueError('!! Model not built yet')
    if not self.launched: self.launch_model(overwrite=False)
    if self.input_type is InputTypes.RNN_BATCH: data = data.as_rnn_batch
    else: assert not data.is_rnn_input
    return data

  # endregion : Private Methods


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import tframe as tfr
from tframe import DataSet
from tframe import hub
from tframe import console
from tframe import pedia

from tframe.enums import InputTypes
from tframe.core import with_graph
from tframe.core import Agent
from tframe.core import TensorSlot, NestedTensorSlot
from tframe.core import SummarySlot, OperationSlot
from tframe.core import Group

from tframe.trainers.metric import Metric
from tframe.trainers.scheme import TrainScheme
from tframe.trainers.trainer import Trainer, TrainerHub
from tframe.trainers.smartrainer import SmartTrainer, SmartTrainerHub


class Model(object):
  """
  Base class of [all?] kinds of models built on TensorFlow
  """
  model_name = 'default'

  def __init__(self, mark=None):
    # Model mark usually helps to decide the folder name
    self.mark = hub.mark or mark
    assert mark is not None

    # Each model has an agent to deal with some tensorflow stuff
    self.agent = Agent(self)

    # Define slots
    self._outputs = TensorSlot(self)

    self._metric = Metric(self, 'metric')
    self._validation_summary = SummarySlot(self)
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
    # Maybe take some notes
    self.agent.take_notes('Model built successfully')
    self.agent.take_notes('Structure:', date_time=False)
    for line in description:
      self.agent.take_notes(line, date_time=False)

  def _build(self, optimizer=None, **kwargs):
    """Abstract method, must be implemented in different models"""
    raise  NotImplementedError('!! build method not implemented')

  def _init_monitor(self):
    if tfr.monitor.activated: tfr.monitor.init_monitor(self)

  @with_graph
  def _define_train_step(self, optimizer=None, var_list=None):
    if not self._loss.activated:
      raise AssertionError('!! loss has not been activated yet')
    with tf.name_scope('Optimizer'):
      if optimizer is None: optimizer = tf.train.AdamOptimizer(1e-4)
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

  def validate_model(self, validation_set, **kwargs):
    assert isinstance(validation_set, DataSet)
    feed_dict = self._get_default_feed_dict(validation_set, is_training=False)
    return self._validate_group.run(feed_dict)

  def take_down_metric(self):
    if not self.metric.activated: return
    notes = 'Record: {:.3f}, Mean Record: {:.3f}'.format(
      self.metric.record, self.metric.mean_record)
    self.agent.take_notes(notes, date_time=False)

  def begin_round(self, **kwargs):
    pass

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

  # endregion : Public Methods

  # region : Private Methods

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


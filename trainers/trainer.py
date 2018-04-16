from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

import tensorflow as tf

from tframe import console
from tframe import TFData
from tframe.core import with_graph
from tframe.models.model import Model
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
    if not isinstance(model, Model):
      raise TypeError('!! model must be an instance of tframe Model')
    self.model = model

    # Date set attributes
    self.training_set = None
    self.validation_set = None
    self.set_data(training_set, validation_set)

    # Set callable attributes
    self._snapshot = self._check_callable(snapshot, 'snapshot')
    self._probe = self._check_callable(probe, 'probe')

    # Initiate trainer hub
    self.th = TrainerHub(self)


  # region : Properties

  @property
  def session(self):
    session = self.model.session
    assert isinstance(session, tf.Session)
    return session

  # endregion : Properties

  # region : Public Methods

  def set_data(self, training_set=None, validation_set=None):
    if training_set is not None:
      self._check_data(training_set, 'training set')
      self.training_set = training_set
    if validation_set is not None:
      self._check_data(validation_set, 'validation set')
      self.validation_set = validation_set

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
    pass


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
      self.training_set.features.shape))
    console.supplement('epochs: {}'.format(self.th.epoch))
    console.supplement('batch size: {}'.format(self.th.batch_size))

  def _check_model(self):
    if self.model.session is None:
      self.model.launch_model(self.th.overwrite)

  # endregion : Before training

  # region : During training

  def _outer_loop(self):
    pass

  def _inner_loop(self):
    pass

  # endregion : During training

  # endregion : Train

  # region : Private Methods

  def _check_data(self, data_set=None, name='dataset'):
    if data_set is None:
      data_set = self.training_set
      name = 'training set'
    if data_set is None: raise ValueError('!! {} not found'.format(name))
    if not isinstance(data_set, TFData):
      raise TypeError('!! {} must be an instance of TFData'.format(name))

  @staticmethod
  def _check_callable(f, name):
    if f is not None and not callable(f):
      raise TypeError('!! {} must be callable'.format(name))
    return f

  # endregion : Private Methods


class TrainerHub(Config):
  """"""
  # :: Define class attributes
  epoch = Flag.integer(1, 'Epoch number to train')
  batch_size = Flag.integer(1, 'Batch size')
  shuffle = Flag.boolean(True, 'Whether to shuffle')

  print_cycle = Flag.integer(0, 'Print cycle')
  validate_cycle = Flag.integer(0, 'Validate cycle')
  snapshot_cycle = Flag.integer(0, 'Snapshot cycle')
  match_cycle = Flag.integer(0, 'Match cycle for RL')

  save_best = Flag.boolean(False, 'Whether to save best')
  idle_tol = Flag.integer(20, 'Torrance of idle rounds when early stop is on')

  def __init__(self, trainer=None):
    self.trainer = trainer

  def set_up(self, **kwargs):
    for key, arg in kwargs.items():
      if hasattr(self, key): self.__setattr__(key, arg)
      else: raise ValueError('!! can not resolve key {}'.format(key))

  def sanity_check(self):
    assert isinstance(self.trainer, Trainer)


# Register trainer hub
TrainerHub.register()



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe import console
from tframe import TFData
from tframe import FLAGS
from tframe.core import with_graph
from tframe.models.model import Model

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

    # Set snapshot
    if snapshot is not None and not callable(snapshot):
      raise TypeError('!! snapshot must be callable')
    self._snapshot = snapshot

    # Set probe
    if probe is not None and not callable(probe):
      raise TypeError('!! probe must be callable')
    self._probe = probe

    # Initiate trainer hub
    self.th = TrainerHub(self)


  # region : Properties

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
  def train(self, config=None, **kwargs):
    # Set trainer hub
    self._init_trainer_hub(config, **kwargs)
    # Do some check-up
    self._check_data(), self._sanity_check(), self.th.sanity_check()

    # Set batch size

    # Train with graph
    pass


  def _init_trainer_hub(self, config, **kwargs):
    if config is not None:
      # If th is provided
      if not isinstance(config, TrainerHub):
        raise TypeError('!! config must be an instance of TrainerHub')
      self.th = config
      self.th.trainer = self
    else: self.th.set_up(**kwargs)

  def _sanity_check(self):
    """Should be overrode by subclasses"""
    pass

  # endregion : Train

  # region : Private Methods

  def _check_data(self, data_set=None, name='dataset'):
    if data_set is None:
      data_set = self.training_set
      name = 'training set'
    if data_set is None: raise ValueError('!! {} not found'.format(name))
    if not isinstance(data_set, TFData):
      raise TypeError('!! {} must be an instance of TFData'.format(name))

  # endregion : Private Methods


class TrainerHub(object):
  def __init__(self, trainer=None):
    self.trainer = trainer

    self.epoch = 1
    self.batch_size = 1
    self.print_cycle = 0
    self.validate_cycle = 0
    self.snapshot_cycle = 0

    self.save_best = False


  def set_up(self, **kwargs):
    # Set epoch
    epoch = kwargs.get('epoch', self.epoch)
    self.epoch = FLAGS.epoch if FLAGS.epoch > 0 else epoch
    # Set batch size
    batch_size = kwargs.get('batch_size', self.batch_size)
    self.batch_size = FLAGS.batch_size if FLAGS.batch_size > 0 else batch_size


  def sanity_check(self):
    assert isinstance(self.trainer, Trainer)



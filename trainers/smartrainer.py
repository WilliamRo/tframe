from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.config import Flag
from tframe.trainers.trainer import Trainer, TrainerHub


class SmartTrainer(Trainer):
  """The so-called 'smart train' refers to automatically tuning learning
      rate and early stopping during training under some criteria"""
  def __init__(
      self,
      model,
      training_set=None,
      validation_set=None,
      snapshot=None,
      probe=None):
    # Call parent's constructor
    Trainer.__init__(
      self, model, training_set, validation_set, snapshot, probe)
    # Override trainer hub with SmartTrainerHub
    self.th = SmartTrainerHub(self)

  # region : Train

  def _sanity_check(self):
    # Smart training relies on model.metric on the validation data set,
    # .. so their existence should be guaranteed
    self._check_data(self._validation_set, 'validation set')
    if getattr(self.model, '_metric', None) is None:
      raise ValueError('!! metric on valid set not defined')

  # endregion : Train


class SmartTrainerHub(TrainerHub):
  """"""
  lr_decay = Flag.float(0.6, 'Learning rate decay coefficient')

  def __init__(self, trainer):
    # Call parent's constructor
    TrainerHub.__init__(self, trainer)
    # Append attributes
    self.ep_count = 0
    self.bad_apples = 0


SmartTrainerHub.register()

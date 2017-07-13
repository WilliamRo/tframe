from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from .. import FLAGS

from .. import config
from .. import console

from ..utils.local import check_path
from ..utils.local import clear_paths

from ..utils.tfdata import TFData


class Model(object):
  """
  Base class of all kinds of models
  """
  def __init__(self, mark=None):
    self.mark = mark

    self._data = None

    self._session = None
    self._summary_writer = None

    self._loss = None
    self._optimizer = None
    self._train_step = None

    self._snapshot = None

  # region : Properties

  @property
  def log_dir(self):
    return check_path(config.record_dir, config.log_folder_name)

  # endregion : Properties

  def train(self, epoch=1, batch_size=128, data=None,
             print_cycle=0, snapshot_cycle=0):
    # Check data
    if data is not None:
      self._data = data
    if self._data is None:
      raise ValueError('Data for training not found')
    elif not isinstance(data, TFData):
      raise TypeError('Data for training must be an instance of TFData')

    # Get epoch and batch size
    epoch = FLAGS.epoch if FLAGS.epoch > 0 else epoch
    batch_size = FLAGS.batch_size if FLAGS.batch_size > 0 else batch_size
    assert isinstance(self._data, TFData)
    self._data.set_batch_size(batch_size)


class Predictor(object):
  def __init__(self):
    self.targets = None

  def predict(self, features):
    pass


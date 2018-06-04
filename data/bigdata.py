from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tframe import console
from tframe import checker
from tframe.utils.local import check_path
from tframe.data.base_classes import TFRData
from tframe.data.dataset import DataSet
from tframe.data.sequences.signals.signal_set import SignalSet


class BigData(TFRData):
  """BigData manages .tfd files in a specific folder."""
  FILE_NAME = 'bigdata.meta'
  EXTENSION = 'meta'

  def __init__(self, data_dir):
    """self.meta = {filename1: data_size_1,
                    filename2: data_size_2
                    ... ...}
    """
    self.meta = {}
    self.data_dir = data_dir
    self.init_method = None
    self._generate_meta(data_dir)

  # region : Properties

  @property
  def structure(self):
    return list(self.meta.values())

  @property
  def size(self):
    return len(self.meta)

  @property
  def is_regular_array(self):
    return False

  # endregion : Properties

  # region : Public Methods

  def get_round_length(self, batch_size, num_steps=None):
    if self.init_method is not None: return None
    round_len = 0
    for len_list in self.structure:
      checker.check_type(len_list, int)
      # Accumulate round length
      if num_steps is None:
        # For feed-forward models
        round_len += int(np.ceil(sum(len_list) / batch_size))
      else:
        # For RNN models
        if num_steps < 0: round_len += len(len_list)
        else: round_len += int(sum([np.ceil(size // batch_size / num_steps)
                                    for size in len_list]))
    # Return round length
    return round_len

  def gen_batches(self, batch_size, shuffle=False):
    for f in self.meta.keys():
      file_path = os.path.join(self.data_dir, f)
      data_set = self._load_data_set(file_path)
      self._check_data_set(data_set)
      for batch in data_set.gen_batches(batch_size, shuffle):
        yield batch

  def gen_rnn_batches(self, batch_size=1, num_steps=-1, shuffle=False):
    for f in self.meta.keys():
      file_path = os.path.join(self.data_dir, f)
      data_set = self._load_data_set(file_path)
      self._check_data_set(data_set)
      for batch in data_set.gen_rnn_batches(batch_size, num_steps, shuffle):
        yield batch

  @classmethod
  def load(cls, data_dir):
    # Check data_dir
    check_path(data_dir, create_path=False)
    # Load or create
    bd_path = os.path.join(data_dir, cls.FILE_NAME)
    if os.path.exists(bd_path):
      bd = super().load(bd_path)
      assert isinstance(bd, BigData)
      bd._check_data_files(data_dir)
    else:
      console.show_status('Metadata not found.')
      bd = BigData(data_dir)
    # Return bigdata
    console.show_status('{} files loaded'.format(len(bd.meta)))
    bd.data_dir = data_dir
    return bd

  # endregion : Public Methods

  # region : Private Methods

  def _check_data_set(self, data_set):
    if callable(self.init_method):
      self.init_method(data_set)
      return
    if isinstance(data_set, SignalSet) and data_set.features is None:
      data_set.init_features_and_targets()

  @staticmethod
  def _load_data_set(file_name):
    assert isinstance(file_name, str)
    extension = file_name.split('.')[-1]
    if extension == DataSet.EXTENSION:
      return DataSet.load(file_name)
    elif extension == SignalSet.EXTENSION:
      return SignalSet.load(file_name)
    else: raise TypeError(
      '!! Can not load file with extension .{}'.format(extension))

  def _get_tfd_list(self, data_dir):
    check_path(data_dir, create_path=False)
    file_list = []
    for f in os.listdir(data_dir):
      file_path = os.path.join(data_dir, f)
      if not os.path.isfile(file_path): continue
      if not 'tfd' in f.split('.')[-1]: continue
      file_list.append(os.path.join(data_dir, f))

    return file_list

  def _check_data_files(self, data_dir):
    file_list = self._get_tfd_list(data_dir)
    console.show_status('Integrity checking ...')
    for i, f in enumerate(file_list):
      if os.path.basename(f) not in self.meta.keys():
        raise AssertionError('!! Can not find {} in metadata'.format(f))
      console.print_progress(i, len(file_list))
    if len(self.meta) != len(file_list):
      raise AssertionError(
        '!! {} files are expected but only {} are found'.format(
          len(self.meta), len(file_list)))

  def _generate_meta(self, data_dir):
    console.show_status('Scanning data directory ...')
    file_list = self._get_tfd_list(data_dir)

    # Scan directory
    num_files = len(file_list)
    for i, file_name in enumerate(file_list):
      data_set = self._load_data_set(file_name)
      self.meta[os.path.basename(file_name)] = data_set.structure
      console.print_progress(i + 1, num_files)
      del data_set

    # Save metadata
    bd_path = os.path.join(data_dir, self.FILE_NAME)
    self.save(bd_path)
    console.show_status('Metadata saved to {}'.format(bd_path))

  # endregion : Private Methods


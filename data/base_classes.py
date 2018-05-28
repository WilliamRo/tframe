from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from tframe import console
from tframe import pedia
import tframe.utils.misc as misc


class TFRData(object):
  """Abstract class defining apis for data set classes used in tframe"""
  
  @property
  def size(self):
    raise NotImplementedError

  def get_round_length(self, batch_size, num_steps=None):
    raise NotImplementedError

  def get_batches(self, batch_size, shuffle=False):
    raise NotImplementedError

  def get_rnn_batches(self, batch_size=1, num_steps=None, shuffle=False):
    raise NotImplementedError


class DataAgent(object):
  """"""
  DATA_NAME = None
  DATA_URL = None
  TFD_FILE_NAME = None

  PROPERTIES = {}

  # region : Properties

  @classmethod
  def default_file_name(cls):
    assert isinstance(cls.DATA_URL, str)
    return cls.DATA_URL.split('/')[-1]

  # endregion : Properties

  # region : Public Methods

  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    """Load (features, targets) as numpy arrays"""
    raise NotImplementedError

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    """Load data as TFrame DataSet"""
    raise NotImplementedError

  @classmethod
  def load(cls, *args, **kwargs):
    """Load data"""
    raise NotImplementedError

  # endregion : Public Methods

  # region : Private Methods

  @classmethod
  def _check_raw_data(cls, data_dir):
    # Get file path
    data_dir = cls._check_path(data_dir, create_path=True)
    file_name = cls.default_file_name()
    file_path = os.path.join(data_dir, file_name)
    # If data does not exist, download from web
    if not os.path.exists(file_path): cls._download(file_path)
    # Return file path
    return file_path

  @classmethod
  def _download(cls, file_path):
    import time
    from six.moves import urllib
    # Show status
    console.show_status('Downloading {} ...'.format(cls.DATA_NAME))
    start_time = time.time()
    def _progress(count, block_size, total_size):
      console.clear_line()
      console.print_progress(count * block_size, total_size, start_time)
    file_path, _ = urllib.request.urlretrieve(
      cls.DATA_URL, file_path, _progress)
    stat_info = os.stat(file_path)
    console.show_status('Successfully downloaded {} ({} bytes).'.format(
      cls.default_file_name(), stat_info.st_size))

  @staticmethod
  def _check_path(*paths, create_path=True):
    assert len(paths) > 0
    if len(paths) == 1:
      paths = re.split(r'/|\\', paths[0])
      if paths[0] in ['.', '']: paths.pop(0)
      if paths[-1] == '': paths.pop(-1)
    path = ""
    for p in paths:
      path = os.path.join(path, p)
      if not os.path.exists(path) and create_path:
        os.mkdir(path)
    # Return path
    return path

  @staticmethod
  def _show_data_sets_info(data_sets):
    from tframe.data.dataset import DataSet
    console.show_status('Data loaded')
    for data_set in data_sets:
      assert isinstance(data_set, DataSet)
      console.supplement('{}:'.format(data_set.name))
      console.supplement(
        'features shape: {}'.format(data_set.features.shape), level=2)
      if data_set.targets is not None:
        console.supplement(
          'targets shape:  {}'.format(data_set.targets.shape), level=2)

  # endregion : Private Methods


class ImageDataAgent(DataAgent):
  """"""
  @classmethod
  def load(cls, data_dir, train_size, validate_size, test_size,
           flatten=False, one_hot=True):
    data_set = cls.load_as_tframe_data(data_dir)
    if flatten:
      data_set.features = data_set.features.reshape(data_set.size, -1)
    if one_hot:
      data_set.targets = misc.convert_to_one_hot(
        data_set.targets, len(data_set['classes']))
    # Split data set
    data_sets = data_set.split(
      train_size, validate_size, test_size,
      names=('training set', 'validation set', 'test set'))
    # Show data info
    cls._show_data_sets_info(data_sets)
    return data_sets



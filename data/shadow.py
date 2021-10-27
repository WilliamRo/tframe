from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tframe.core.nomear import Nomear


class DataShadow(Nomear):

  DATA_KEY = 'data'

  _global_root = None
  _queue = []
  _max_size = None

  def __init__(self, data_path=None, load_method=None):
    self.data_path = data_path
    self.load_method = load_method


  @property
  def data(self):
    return self.get_from_pocket(self.DATA_KEY, initializer=self._load_data)


  @property
  def data_root(self):
    if isinstance(self._global_root, str) and os.path.isdir(self._global_root):
      return self._global_root

    from tframe import hub
    return hub.data_dir


  @classmethod
  def set_data_root(cls, path: str):
    assert os.path.isdir(path)
    cls._global_root = path


  @classmethod
  def set_max_size(cls, val: int):
    assert val > 0
    cls._max_size = val
    print(f'>> Max size of shadow list has been set to {val}')


  @classmethod
  def check_memory(cls):
    if cls._max_size is None or len(cls._queue) <= cls._max_size: return
    d: DataShadow = cls._queue.pop(0)
    d._pocket.pop(cls.DATA_KEY)


  def _load_data(self):
    # If load_method is provided, use this function directly
    if callable(self.load_method): return self.load_method()
    # Otherwise load data according to its path
    # Currently only .jpg data is supported
    path: str = self.data_path
    assert path.endswith('.jpg')

    # Load data using PIL.Image
    from PIL import Image
    img = np.array(Image.open(os.path.join(self.data_root, path)))

    # Register data
    self._queue.append(self)
    # Release queue head if necessary
    self.check_memory()

    return img

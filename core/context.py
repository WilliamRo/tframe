from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import OrderedDict

from tframe.configs.config_base import Config
from tframe.monitor import Monitor


# Public methods

class Context(object):
  """The author is too lazy to add a description"""
  def __init__(self):
    # Private attribute
    self._center_od_ = OrderedDict()
    self._current_graph = None
    self._trainer = None

    # Public attribute
    self.hub = Config()
    self.monitor = Monitor()

  # region : Properties

  @property
  def current_graph(self):
    assert isinstance(self._current_graph, tf.Graph)
    return self._current_graph

  @current_graph.setter
  def current_graph(self, val):
    assert self._current_graph is None and isinstance(val, tf.Graph)
    self._current_graph = val

  @property
  def trainer(self):
    from tframe.trainers.trainer import Trainer
    assert isinstance(self._trainer, Trainer)
    return self._trainer

  @trainer.setter
  def trainer(self, val):
    from tframe.trainers.trainer import Trainer
    assert self._trainer is None and isinstance(val, Trainer)
    self._trainer = val

  # endregion : Properties

  # region : Public Methods

  def add_to_list_collection(self, name, val):
    collection = self.get_collection_by_key(name, True, list)
    collection.append(val)

  def add_to_dict_collection(self, name, key, val, atomic=True):
    collection = self.get_collection_by_key(name, True, dict)
    if atomic: assert key not in collection.keys()
    collection[key] = val

  def add_collection(self, key, val):
    assert key not in self._center_od_.keys()
    self._center_od_[key] = val

  def get_collection_by_key(self, name, init_if_necessary=False, val_type=None):
    if val_type is not None:
      assert val_type in (set, list, tuple, dict, OrderedDict)
    if init_if_necessary:
      assert val_type is not None
      if val_type is dict: val_type = OrderedDict
      if name not in self._center_od_.keys():
        self._center_od_[name] = val_type()
    val = self._center_od_[name]
    if val_type is not None: assert isinstance(val, val_type)
    return val

  def write_value(self, key, val):
    self.add_collection(key, val)

  def read_value(self, key, **kwargs):
    if key not in self._center_od_.keys():
      if 'default_value' in kwargs.keys():
        return kwargs.get('default_value')
      else: raise KeyError('!! Value with key `{}` not found.'.format(key))
    else: return self._center_od_[key]

  # endregion : Public Methods


# Initiate a context
context = Context()
hub = context.hub
monitor = context.monitor






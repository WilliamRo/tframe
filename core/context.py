from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import OrderedDict

from tframe.configs.config_base import Config
from tframe.utils.monitor import Monitor
# from tframe.deprecated.monitor import Monitor
import tframe.utils.checker as checker


# Public methods

class Context(object):
  """The author is too lazy to add a description"""
  _LOSSES_LIST = '_LOSSES_LIST'
  _TENSORS_EXPORT_DICT = '_TENSOES_EXPORT_DICT'
  _VARIABLE_EXPORT_DICT = '_VARIABLE_EXPORT_DICT'

  S_IN_DYDS = 'S_IN_DYDS'

  def __init__(self):
    # Private attribute
    self._center_od_ = OrderedDict()
    self._current_graph = None
    self._trainer = None

    # Public attribute
    self.hub = Config()
    self.monitor = Monitor()

    # Nailed attributes
    self.customed_loss_f_net = None
    self.customed_outer_loss_f_net = None

    # Loss function (will be set only in Predictor.build and be used in
    #   RNet._link)
    self.loss_function = None

    # Global variables
    self.metric_name = 'Metric'  # TODO: who will use it?
    self.logits_tensor_dict = {}
    self.reuse_dict = OrderedDict()

    # pruner will be initiated in the early stage of model building
    self.pruner = None

    # Sparse tensor list
    self.weights_list = []
    self.sparse_weights_list = []

    # Note
    self.note = None

    # This placeholder is used for sequence classification
    self.gather_indices = None

    # Optimizer list for resurrection
    self.tf_optimizer = None

    # Counter for shortcuts
    self._short_cut_counter = 0

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

  @property
  def loss_tensor_list(self):
    return self.get_collection_by_key(
      self._LOSSES_LIST, init_if_necessary=True, val_type=list)

  @property
  def tensors_to_export(self):
    return self.get_collection_by_key(
      self._TENSORS_EXPORT_DICT, init_if_necessary=True, val_type=dict)

  @property
  def variables_to_export(self):
    return self.get_collection_by_key(
      self._VARIABLE_EXPORT_DICT, init_if_necessary=True, val_type=dict)

  # endregion : Properties

  # region : Public Methods

  def get_next_output_id(self):
    self._short_cut_counter += 1
    return self._short_cut_counter

  def has_collection(self, key):
    assert isinstance(key, str)
    return key in self._center_od_.keys()

  def add_to_list_collection(self, name, val):
    collection = self.get_collection_by_key(name, True, list)
    collection.append(val)

  def add_to_dict_collection(self, name, key, val):
    collection = self.get_collection_by_key(name, True, dict)
    # Check key
    if key in collection.keys(): key += '_{}'.format(
      2 + len([k for k in collection.keys() if key + '_' in k]))
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

  def has_value(self, key):
    return self.has_collection(key)

  def clear_all_collections(self):
    """This method will only be called inside Recurrent._build_while_free
       to ensure tensors inside while_loop are correctly handle.
    """
    self._center_od_.clear()
    self.logits_tensor_dict = {}
    # Clear pruner
    if self.pruner is not None: self.pruner.clear()
    # Clear shortcut counter
    self._short_cut_counter = 0

  # endregion : Public Methods

  # region : Collection short cuts

  def add_loss_tensor(self, loss):
    assert isinstance(loss, tf.Tensor)
    checker.check(len(loss.shape) == 0, 'Input tensor must be a scalar')
    self.add_to_list_collection(self._LOSSES_LIST, loss)

  def add_var_to_export(self, name, var):
    assert isinstance(name, str)
    self.add_to_dict_collection(self._VARIABLE_EXPORT_DICT, name, var)

  def add_tensor_to_export(self, name, tensor):
    assert isinstance(name, str)
    self.add_to_dict_collection(self._TENSORS_EXPORT_DICT, name, tensor)

  def get_logits(self, output):
    """Currently RNNs do not support logits
       Logits will be used in (1) here (2) net (3) quantities
    """
    logits = self.logits_tensor_dict.get(output, None)
    # TODO: This is a compromise for RNN with single logits tensor
    if logits is None and len(self.logits_tensor_dict) == 1:
      logits = list(self.logits_tensor_dict.values())[0]
    return logits

  def set_rnn_logits(self, logits):
    # TODO: This is a compromise for RNN with single logits tensor
    # After running this, logits can be properly found in quantity.__call__
    assert len(self.logits_tensor_dict) == 1
    key = list(self.logits_tensor_dict.keys())[0]
    self.logits_tensor_dict[key] = logits

  # endregion : Collection short cuts

  # region : MISC

  def set_logits_tensor(self, output, logits):
    self.logits_tensor_dict[output] = logits
  # region : Public Static Methods

  @staticmethod
  def open_input_port(name, input_shape, dtype=None, add_batch_dim=True):
    """Create and add an input placeholder into the default tf collection.
    TODO: the placeholder should be added to a model-related collection."""
    from tframe import hub, pedia  # TODO refactor this
    # Sanity check
    assert isinstance(input_shape, (tuple, list))
    if add_batch_dim: input_shape = [None] + list(input_shape)
    if dtype is None: dtype = hub.dtype
    input_ = tf.placeholder(dtype=dtype, shape=input_shape, name=name)
    # Add to collection
    tf.add_to_collection(pedia.default_feed_dict, input_)
    # Return placeholder
    return input_

  # endregion : Public Static Methods


  # endregion : MISC


# Initiate a context
context = Context()
hub = context.hub
monitor = context.monitor






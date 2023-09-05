from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
from tframe import tf
from collections import OrderedDict

import tframe as tfr
from tframe.core.nomear import Nomear

from .flag import Flag
from .advanced_configs import AdvancedConfigs
from .cloud_configs import CloudConfigs
from .dataset_configs import DataConfigs
from .display_configs import DisplayConfig
from .model_configs import ModelConfigs
from .monitor_configs import MonitorConfigs
from .note_configs import NoteConfigs
from .rnn_configs import RNNConfigs
from .trainer_configs import TrainerConfigs


class Config(
  AdvancedConfigs,
  CloudConfigs,
  DataConfigs,
  DisplayConfig,
  ModelConfigs,
  MonitorConfigs,
  NoteConfigs,
  TrainerConfigs,
  RNNConfigs,
  Nomear,
):
  registered = False

  record_dir = Flag.string('records', 'Root path for records')
  log_folder_name = Flag.string('logs', '...')
  ckpt_folder_name = Flag.string('checkpoints', '...')
  snapshot_folder_name = Flag.string('snapshots', '...')

  job_dir = Flag.string(
    './records', 'The root directory where the records should be put',
    name='job-dir')
  data_dir = Flag.string('', 'The data directory')
  raw_data_dir = Flag.string(None, 'Raw data directory')

  dtype = Flag.whatever(tf.float32, 'Default dtype for tensors', is_key=None)
  tb_port = Flag.integer(6006, 'Tensorboard port number')
  show_structure_detail = Flag.boolean(True, '...')

  # logging will be suppressed if this flag is set to True when agent
  #   is launching a model
  suppress_logging = Flag.boolean(
    True, 'Whether to set logging level down to get rid of the device '
          'information')
  progress_bar = Flag.boolean(True, 'Whether to show progress bar')

  keep_trainer_log = Flag.boolean(
    False, 'Whether to keep trainer logs. Usually be used for probe '
           'methods')

  # Device related config
  visible_gpu_id = Flag.string(
    None, 'CUDA_VISIBLE_DEVICES option', name='gpu_id')
  allow_growth = Flag.boolean(True, 'tf.ConfigProto().gpu_options.allow_growth')
  gpu_memory_fraction = Flag.float(
    0.4, 'config.gpu_options.per_process_gpu_memory_fraction')

  # Other fancy stuff
  int_para_1 = Flag.integer(None, 'Used to pass an integer parameter using '
                               ' command line')
  bool_para_1 = Flag.boolean(False, 'Used to pass a boolean parameter using'
                                    ' command line')
  alpha = Flag.float(None, 'Alpha', is_key=None)
  beta = Flag.float(None, 'Beta', is_key=None)
  gamma = Flag.float(None, 'Gamma', is_key=None)
  epsilon = Flag.float(None, 'Epsilon', is_key=None)
  delta = Flag.float(None, 'Delta', is_key=None)

  developer_code = Flag.string('', 'Code for developers to debug', is_key=None)
  developer_args = Flag.string(
    '', 'Args for developers to develop', is_key=None)
  verbose_config = Flag.string('', 'String for configuring verbosity')

  stats_max_length = Flag.integer(20, 'Maximum length a Statistic can keep')

  allow_activation = Flag.boolean(True, 'Whether to allow activation')
  visualize_tensors = Flag.boolean(
    False, 'Whether to visualize tensors in core')
  visualize_kernels = Flag.boolean(
    False, 'Whether to visualize CNN kernels in core')

  tensor_dict = Flag.whatever(None, 'Stores tensors for visualization')

  tic_toc = Flag.boolean(False, 'Whether to track time')

  depot = Flag.whatever({}, 'You can put whatever things here')

  # A dictionary for highest priority setting
  _backdoor = {}

  def __init__(self, as_global=False):
    # Try to register flags into tensorflow
    if not self.__class__.registered:
      self.__class__.register()

    if as_global:
      # TODO:
      tfr.hub.__class__ = self.__class__
      # tfr.hub = self
      # tfr.context.hub = self
      tfr.hub.redirect(self)

  # region : Properties

  @property
  def use_dynamic_ground_truth(self):
    return callable(self.dynamic_ground_truth_generator)

  @property
  def should_create_path(self):
    if tfr.hub.rehearse: return True
    return (self.train or self.dynamic_evaluation) and not self.on_cloud

  @property
  def np_dtype(self):
    if self.dtype is tf.float32: return np.float32
    elif self.dtype is tf.float64: return np.float64
    elif self.dtype is tf.float16: return np.float16
    raise ValueError

  @property
  def key_options(self):
    ko = OrderedDict()
    for name in self.__dir__():
      if name in ('key_options', 'config_strings'): continue
      attr = self.get_attr(name)
      if not isinstance(attr, Flag): continue
      if attr.is_key:
        ko[name] = attr.value
    return ko

  @property
  def config_strings(self):
    return sorted(['{}: {}'.format(k, v) for k, v in self.key_options.items()])

  @property
  def developer_options(self):
    if not self.developer_args: return None
    from tframe.utils.arg_parser import Parser
    parser = Parser.parse(self.developer_args)
    return parser

  # endregion : Properties

  # region : Override

  def __getattribute__(self, name):
    attr = object.__getattribute__(self, name)
    if not isinstance(attr, Flag): return attr
    else:
      if name in self._backdoor: return self._backdoor[name]
      return attr.value

  def __setattr__(self, name, value):
    # Set value to backdoor if required, e.g., th.batch_size = {[256]}
    if isinstance(value, list) and len(value) == 1:
      if isinstance(value[0], set) and len(value[0]) == 1:
        self._backdoor[name] = list(value[0])[0]

    # If attribute is not found (say during instance initialization),
    # .. use default __setattr__
    if not hasattr(self, name):
      object.__setattr__(self, name, value)
      return

    # If attribute is not a Flag, use default __setattr__
    attr = object.__getattribute__(self, name)
    if not isinstance(attr, Flag):
      object.__setattr__(self, name, value)
      return

    # Now attr is definitely a Flag
    # if name == 'visible_gpu_id':
    #   import os
    #   assert isinstance(value, str)
    #   os.environ['CUDA_VISIBLE_DEVICES'] = value

    if attr.frozen and value != attr._value:
      raise AssertionError(
        '!! config {} has been frozen to {}'.format(name, attr._value))
    # If attr is a enum Flag, make sure value is legal
    if attr.is_enum:
      if value not in list(attr.enum_class):
        raise TypeError(
          '!! Can not set {} for enum flag {}'.format(value, name))

    attr._value = value
    if attr.ready_to_be_key: attr._is_key = True

    # Replace the attr with a new Flag TODO: tasks with multi hubs?
    # object.__setattr__(self, name, attr.new_value(value))


  # endregion : Override

  # region : Public Methods

  def config_dir(self, dir_depth=2):
    """This method should be called only in XX_core.py module for setting
       default job_dir and data_dir.
    """
    self.job_dir = os.path.join(sys.path[dir_depth - 1])
    self.data_dir = os.path.join(self.job_dir, 'data')
    tfr.console.show_status('Job directory set to `{}`'.format(self.job_dir))

  def update_job_dir(self, id, model_name):
    """A quick API"""
    from tframe.utils.organizer.task_tools import update_job_dir
    return update_job_dir(id, model_name, fs_index=-3)

  def set_date_as_prefix(self):
    from tframe.utils.misc import date_string
    self.prefix = '{}_'.format(date_string())

  @staticmethod
  def decimal_str(num, decimals=3):
    assert np.isscalar(num)
    assert isinstance(decimals, int) and decimals > 0
    fmt = '{:.' + str(decimals) + 'f}'
    return fmt.format(num)

  @classmethod
  def register(cls):
    queue = {key: getattr(cls, key) for key in dir(cls)
             if isinstance(getattr(cls, key), Flag)}
    for name, flag in queue.items():
      if flag.should_register: flag.register(name)
      elif flag.name is None: flag.name = name
    cls.registered = True

  def redirect(self, config):
    """Redirect self to config"""
    assert isinstance(config, Config)

    # flag_names = [name for name, value in self.__dict__.items()
    #               if isinstance(value, Flag)]

    flag_names = [name for name in config.__dir__()
                  if hasattr(config, name) and
                  isinstance(object.__getattribute__(config, name), Flag)]
    for name in flag_names:
      # value = getattr(config, name)
      # Set flag to self

      object.__setattr__(self, name, config.get_flag(name))

      # Assign value to self.flag
      # self.__setattr__(name, value)

  def smooth_out_conflicts(self):
    self.smooth_out_cloud_configs()
    self.smooth_out_monitor_configs()
    self.smooth_out_note_configs()
    self.smooth_out_model_configs()
    self.smooth_out_trainer_configs()

    if self.export_dl_dx or self.export_dl_ds_stat:
      # TODO: these 2 options should be used carefully,
      #       since sequences with different lengths may yield
      #       incorrect result
      self.allow_loss_in_loop = True
    if self.prune_on and self.pruning_iterations > 0:
      self.overwrite = False
    if self.export_weight_grads:
      # TODO: These 2 configs could be merged as one
      self.monitor_weight_grads = True
    if self.etch_on:
      self.monitor_weight_grads = True

  def get_attr(self, name):
    return object.__getattribute__(self, name)

  def get_flag(self, name):
    flag = super().__getattribute__(name)
    if not isinstance(flag, Flag):
      raise TypeError('!! flag {} not found'.format(name))
    return flag

  def get_optimizer(self, optimizer=None):
    """Get tframe optimizer (based on tensorflow optimizer). """
    from tframe.optimizers.optimizer import Optimizer

    if optimizer is None:
      if any([self.optimizer is None, self.learning_rate is None]):
        tfr.console.show_status('Optimizer not defined.', '!!')
        return None

      optimizer = self.optimizer
      tfr.console.show_status(
        'Optimizer defined in trainer hub is used.', '++')

    return Optimizer.get_optimizer(optimizer)

  # endregion : Public Methods


Config.register()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tframe as tfr
from tframe.enums import EnumPro

flags = tf.app.flags

# TODO: Value set to Flag should be checked

class Flag(object):
  def __init__(self, default_value, description, register=None, name=None,
               is_key=False, **kwargs):
    """
    ... another way to design this class is to let name attribute be assigned
    when registered, but we need to allow FLAGS names such as 'job-dir' to be
    legal. In this perspective, this design is better.
    :param name:
    :param default_value:
    :param description:
    :param is_key: (1) True: force to show;
                    (2) False: force to hide;
                    (3) None: show if has been modified
    :param register: if is None, this flag can not be passed via tf FLAGS
    """
    self._default_value = default_value
    self._description = description
    self._register = register
    self._name = name
    self._is_key = is_key
    self._kwargs = kwargs

    self._value = default_value
    self._frozen = False

  # region : Properties

  @property
  def ready_to_be_key(self):
    return self._is_key is None

  @property
  def is_key(self):
    if self._is_key is False: return False
    if not self.should_register: return self._is_key is True
    assert hasattr(flags.FLAGS, self._name)
    return self._is_key is True or getattr(flags.FLAGS, self._name) is not None

  @property
  def frozen(self):
    return self._frozen

  @property
  def value(self):
    # If not registered to tf.app.flags or has been frozen
    if self._register is None or self._frozen: return self._value

    assert hasattr(flags.FLAGS, self._name)
    f_value = getattr(flags.FLAGS, self._name)
    # Configs defined via tensorflow FLAGS have priority over any other way
    if f_value is None: return self._value
    # If self is en enum Flag, then f_value must be a string in
    # .. self.enum_class.value_list(), so we need to get its member
    if self.is_enum: f_value = self.enum_class.get_member(f_value)
    if self.frozen and self._value != f_value:
      raise AssertionError(
        "!! Invalid tensorflow FLAGS value {0}={1} 'cause {0} has been "
        "frozen to {2}".format(self._name, f_value, self._value))
    return f_value

  @property
  def should_register(self):
    return self._register is not None

  @property
  def enum_class(self):
    cls = self._kwargs.get('enum_class', None)
    if cls is None or not issubclass(cls, EnumPro): return None
    return cls

  @property
  def is_enum(self):
    return self.enum_class is not None and self._register is flags.DEFINE_enum

  # endregion : Properties

  # region : Class Methods

  @classmethod
  def whatever(cls, default_value, description, is_key=False):
    return Flag(default_value, description, is_key=is_key)

  @classmethod
  def string(cls, default_value, description, name=None, is_key=False):
    return Flag(default_value, description, flags.DEFINE_string, name,
                is_key=is_key)

  @classmethod
  def boolean(cls, default_value, description, name=None, is_key=False):
    return Flag(default_value, description, flags.DEFINE_boolean, name,
                is_key=is_key)

  @classmethod
  def integer(cls, default_value, description, name=None, is_key=False):
    return Flag(default_value, description, flags.DEFINE_integer, name,
                is_key=is_key)

  @classmethod
  def float(cls, default_value, description, name=None, is_key=False):
    return Flag(default_value, description, flags.DEFINE_float, name,
                is_key=is_key)

  @classmethod
  def list(cls, default_value, description, name=None):
    return Flag(default_value, description, flags.DEFINE_list, name)

  @classmethod
  def enum(cls, default_value, enum_class, description, name=None,
           is_key=False):
    assert issubclass(enum_class, EnumPro)
    return Flag(default_value, description, flags.DEFINE_enum, name,
                enum_class=enum_class, is_key=is_key)

  # endregion : Class Methods

  # region : Public Methods

  def register(self, name):
    # If name is not specified during construction, use flag's attribute name
    # .. in Config
    if self._name is None: self._name = name
    if self._register is None or self._name in list(flags.FLAGS): return
    # Register enum flag
    if self.is_enum:
      flags.DEFINE_enum(
        self._name, None, self.enum_class.value_list(), self._description)
      return
    # Register other flag
    assert self._register is not flags.DEFINE_enum
    self._register(self._name, None, self._description)

  def new_value(self, value):
    flg = Flag(self._default_value, self._description, self._register,
               self._name, **self._kwargs)
    flg._value = value
    return flg

  def freeze(self, value):
    self._value = value
    self._frozen = True

  # endregion : Public Methods


class Config(object):
  """"""
  # :: Define class attributes
  # Old config
  record_dir = Flag.string('records', 'Root path for records')
  note_folder_name = Flag.string('notes', '...')
  log_folder_name = Flag.string('logs', '...')
  ckpt_folder_name = Flag.string('checkpoints', '...')
  snapshot_folder_name = Flag.string('snapshots', '...')
  gather_file_name = Flag.string('gather.txt', '...')
  gather_summ_name = Flag.string('gather.sum', '...')
  tb_port = Flag.integer(6006, 'Tensorboard port number')
  show_structure_detail = Flag.boolean(False, '...')

  block_validation = Flag.whatever(False, '???')
  dtype = Flag.whatever(tf.float32, 'Default dtype for tensors', is_key=None)

  visible_gpu_id = Flag.string(
    None, 'CUDA_VISIBLE_DEVICES option', name='gpu_id')

  # Migrated from tframe\__init__.py
  train = Flag.boolean(True, 'Whether this is a training task')
  parallel_on = Flag.boolean(False, 'Whether to turn on parallel option')
  smart_train = Flag.boolean(False, 'Whether to use smart trainer', is_key=None)
  save_model = Flag.boolean(True, 'Whether to save model during training')
  overwrite = Flag.boolean(False, 'Whether to overwrite records')
  export_note = Flag.boolean(False, 'Whether to take notes')
  summary = Flag.boolean(True, 'Whether to write summary')
  epoch_as_step = Flag.boolean(True, '...')
  snapshot = Flag.boolean(False, 'Whether to take snapshot during training')
  job_dir = Flag.string(
    './records', 'The root directory where the records should be put',
    name='job-dir')
  data_dir = Flag.string('', 'The data directory')

  # logging will be suppressed if this flag is set to True when agent
  #   is launching a model
  suppress_logging = Flag.boolean(
    True, 'Whether to set logging level down to get rid of the device '
          'information')
  progress_bar = Flag.boolean(True, 'Whether to show progress bar')
  on_cloud = Flag.boolean(
    False, 'Whether this task is running on the cloud',
    name='cloud')
  hp_tuning = Flag.boolean(
    False, 'Whether this is a hyper-parameter tuning task',
    name='hpt')
  rand_over_classes = Flag.boolean(False, '...', is_key=None)
  keep_trainer_log = Flag.boolean(False, 'Whether to keep trainer logs.')
  auto_gather = Flag.boolean(
    False, 'If set to True, agent will gather information in a default way'
           ' when export_note flag is set to True')
  export_note_to_summ = Flag.boolean(False, 'Whether to export note summary')

  # Monitor options
  monitor = Flag.boolean(None, 'Whether to monitor or not (of highest '
                               'priority)')
  monitor_grad = Flag.boolean(False, 'Whether to monitor gradients or not')
  monitor_weight = Flag.boolean(False, 'Whether to monitor weights or not')
  monitor_preact = Flag.boolean(False, 'Whether to enable pre-act summary')
  monitor_postact = Flag.boolean(False, 'Whether to enable post-act summary')

  # Device related config
  allow_growth = Flag.boolean(True, 'tf.ConfigProto().gpu_options.allow_growth')
  gpu_memory_fraction = Flag.float(
    0.4, 'config.gpu_options.per_process_gpu_memory_fraction')

  # Configs usually provided during method calling
  mark = Flag.string(None, 'Model identifier', is_key=True)
  suffix = Flag.string(None, 'Suffix to mark')
  model = Flag.whatever(None, 'A function which returns a built model')
  learning_rate = Flag.float(None, 'Learning rate', name='lr', is_key=None)
  regularizer = Flag.string('l2', 'Regularizer', name='reg', is_key=None)
  reg_strength = Flag.float(0.0, 'Regularizer strength', name='reg_str',
                            is_key=None)
  weight_initializer = Flag.whatever(None, 'Weight initializer')
  bias_initializer = Flag.whatever(None, 'Bias initializer')
  actype1 = Flag.string('relu', 'Activation type 1', is_key=None)
  actype2 = Flag.string('relu', 'Activation type 2', is_key=None)
  input_gate = Flag.boolean(True, 'Whether to use input gate in LSTM',
                            is_key=None)
  forget_gate = Flag.boolean(True, 'Whether to use forget gate in LSTM',
                             is_key=None)
  output_gate = Flag.boolean(True, 'Whether to use output gate in LSTM',
                             is_key=None)
  use_bias = Flag.boolean(True, 'Whether to use bias', is_key=None)
  fc_memory = Flag.boolean(True, 'Whether to fully connect memory', is_key=None)
  act_memory = Flag.boolean(True, 'Whether to activate memory', is_key=None)
  val_preheat = Flag.integer(0, 'metric = metric_batch[val_preheat:].mean')
  val_batch_size = Flag.integer(None, 'Batch size in batch validation')
  with_peepholes = Flag.boolean(False, 'Whether to add peepholes in LSTM',
                                is_key=None)
  neurons_per_unit = Flag.integer(3, '...', is_key=None)
  hidden_dim = Flag.integer(-1, 'Hidden dimension', is_key=None)
  fc_dims = Flag.whatever(None, '...')
  rc_dims = Flag.whatever(None, '...')
  num_blocks = Flag.integer(-1, 'Block number in model', is_key=None)
  input_shape = Flag.list([], 'Input shape of samples')
  output_dim = Flag.integer(0, 'Output dimension of a model')
  num_classes = Flag.integer(-1, 'Class number for classification tasks')
  memory_depth = Flag.integer(1, 'Memory depth for system identification')
  loss_function = Flag.whatever('cross_entropy', 'Loss function')
  notify_when_reset = Flag.whatever(False, '...')
  optimizer = Flag.whatever(None, 'optimizer')

  # Advanced RNN option
  truncate_grad = Flag.boolean(None, 'Whether to truncate gradient in RNN',
                               is_key=None)
  forward_gate = Flag.boolean(
    None, 'Whether to calculate units using gate units from previous time '
          'step', is_key=None)
  mem_cfg = Flag.string('', 'e.g. `7-a-f;8-na-nf`', is_key=None)

  # Other fancy stuff
  show_record_history_in_note = Flag.boolean(False, '...')
  # TODO: temporarily be put here cuz agent may access to them
  note_cycle = Flag.integer(0, 'Note cycle')
  note_per_round = Flag.integer(0, 'Note per round')

  # Shelter
  sample_num = Flag.integer(9, 'Sample number in some unsupervised learning '
                               'tasks')
  int_para_1 = Flag.integer(0, 'Used to pass an integer parameter using '
                               ' command line')
  bool_para_1 = Flag.boolean(False, 'Used to pass a boolean parameter using'
                                    ' command line')

  # BETA:
  use_rtrl = Flag.boolean(
    False, 'Whether to use RTRL in training RNN', is_key=None)
  test_grad = Flag.boolean(False, '...')

  def __init__(self, as_global=False):
    if as_global:
      tfr.hub.redirect(self)

  # region : Properties

  @property
  def should_create_path(self):
    return self.train and not self.on_cloud

  @property
  def key_options(self):
    ko = {}
    for name in self.__dir__():
      if name in ('key_options', 'config_strings'): continue
      attr = self.get_attr(name)
      if not isinstance(attr, Flag): continue
      if attr.is_key:
        ko[name] = attr.value

    return ko

  @property
  def config_strings(self):
    return ['{}: {}'.format(k, v) for k, v in self.key_options.items()]

    # css = []
    # for name in self.__dir__():
    #   if name == 'config_strings': continue
    #   attr = self.get_attr(name)
    #   if not isinstance(attr, Flag): continue
    #   if attr.is_key:
    #     css.append('{}: {}'.format(name, attr.value))
    #
    # return css

  # endregion : Properties

  # region : Override

  def __getattribute__(self, name):
    attr = object.__getattribute__(self, name)
    if not isinstance(attr, Flag): return attr
    else: return attr.value

  def __setattr__(self, name, value):
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

  @classmethod
  def register(cls):
    queue = {name: value for name, value in cls.__dict__.items()
             if isinstance(value, Flag) and value.should_register}
    for name, flg in queue.items(): flg.register(name)

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
    """"""
    if '://' in self.job_dir: self.on_cloud = True
    if self.on_cloud or self.hp_tuning:
      self.export_note = False
      self.progress_bar = False
    if self.on_cloud:
      self.snapshot = False
      self.monitor = False
    if self.hp_tuning:
      self.summary = False
      self.save_model = False
    if not self.train and self.on_cloud: self.overwrite = False

    if self.monitor in (True, False):
      self.monitor_grad = self.monitor
      self.monitor_weight = self.monitor
      self.monitor_preact = self.monitor
      self.monitor_postact = self.monitor

  def get_attr(self, name):
    return object.__getattribute__(self, name)

  def get_flag(self, name):
    flag = super().__getattribute__(name)
    if not isinstance(flag, Flag):
      raise TypeError('!! flag {} not found'.format(name))
    return flag

  # endregion : Public Methods

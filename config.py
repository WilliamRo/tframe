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
               **kwargs):
    """
    ... another way to design this class is to let name attribute be assigned
    when registered, but we need to allow FLAGS names such as 'job-dir' to be
    legal. In this perspective, this design is better.
    :param name:
    :param default_value:
    :param description:
    :param register: if is None, this flag can not be passed via tf FLAGS
    """
    self._default_value = default_value
    self._description = description
    self._register = register
    self._name = name
    self._kwargs = kwargs

    self._value = default_value
    self._frozen = False

  # region : Properties

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
    if self.is_enum: return self.enum_class.get_member(f_value)
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
  def whatever(cls, default_value, description):
    return Flag(default_value, description)

  @classmethod
  def string(cls, default_value, description, name=None):
    return Flag(default_value, description, flags.DEFINE_string, name)

  @classmethod
  def boolean(cls, default_value, description, name=None):
    return Flag(default_value, description, flags.DEFINE_boolean, name)

  @classmethod
  def integer(cls, default_value, description, name=None):
    return Flag(default_value, description, flags.DEFINE_integer, name)

  @classmethod
  def float(cls, default_value, description, name=None):
    return Flag(default_value, description, flags.DEFINE_float, name)

  @classmethod
  def list(cls, default_value, description, name=None):
    return Flag(default_value, description, flags.DEFINE_list, name)

  @classmethod
  def enum(cls, default_value, enum_class, description, name=None):
    assert issubclass(enum_class, EnumPro)
    return Flag(default_value, description, flags.DEFINE_enum, name,
                enum_class=enum_class)

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
  log_folder_name = Flag.string('logs', '...')
  ckpt_folder_name = Flag.string('checkpoints', '...')
  snapshot_folder_name = Flag.string('snapshots', '...')

  block_validation = Flag.whatever(False, '???')
  dtype = Flag.whatever(tf.float32, 'Default dtype for tensors')

  # Migrated from tframe\__init__.py
  summary = Flag.boolean(True, 'Whether to write summary')
  save_model = Flag.boolean(True, 'Whether to save model during training')
  snapshot = Flag.boolean(False, 'Whether to take snapshot during training')
  train = Flag.boolean(True, 'Whether this is a training task')
  smart_train = Flag.boolean(False, 'Whether to use smart trainer')
  overwrite = Flag.boolean(False, 'Whether to overwrite records')
  job_dir = Flag.string(
    './', 'The root directory where the records should be put',
    name='job-dir')
  data_dir = Flag.string('', 'The data directory')

  activation_sum = Flag.boolean(
    False, 'Whether to enable activation summary',
    name='act_sum')
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

  # Configs usually provided during method calling
  mark = Flag.string(None, 'Model identifier')
  learning_rate = Flag.float(None, 'Learning rate', name='lr')

  # Shelter
  sample_num = Flag.integer(9, 'Sample number in some unsupervised learning '
                               'tasks')

  def __init__(self, as_global=False):
    if as_global:
      tfr.hub.redirect(self)

  # region : Properties

  @property
  def should_create_path(self):
    return self.train and not self.on_cloud

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
    if attr.frozen:
      raise AssertionError('!! config {} has been frozen'.format(name))
    # If attr is a enum Flag, make sure value is legal
    if attr.is_enum:
      if value not in list(attr.enum_class):
        raise TypeError(
          '!! Can not set {} for enum flag {}'.format(value, name))

    attr._value = value

    # Replace the attr with a new Flag TODO: why?
    # object.__setattr__(self, name, attr.new_value(value))


  # endregion : Override

  # region : Public Methods

  @classmethod
  def register(cls):
    queue = {name: value for name, value in cls.__dict__.items()
             if isinstance(value, Flag) and value.should_register}
    for name, flg in queue.items(): flg.register(name)

  def redirect(self, config):
    assert isinstance(config, Config)
    flag_names = [name for name, value in self.__dict__.items()
                  if isinstance(value, Flag)]
    for name in flag_names:
      object.__setattr__(self, name, config.get_flag(name))


  def smooth_out_conflicts(self):
    """"""
    if '://' in self.job_dir: self.on_cloud = True
    if self.on_cloud:
      self.progress_bar = False
      self.snapshot = False
    if self.hp_tuning:
      self.summary = False
      self.save_model = False
    if not self.train and self.on_cloud: self.overwrite = False

  def get_attr(self, name):
    return object.__getattribute__(self, name)

  def get_flag(self, name):
    flag = super().__getattribute__(name)
    if not isinstance(flag, Flag):
      raise TypeError('!! flag {} not found'.format(name))
    return flag

  # endregion : Public Methods

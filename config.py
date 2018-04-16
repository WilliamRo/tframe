from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Flag(object):
  def __init__(self, default_value, description, register=None, name=None):
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

    self._value = default_value

  # region : Properties

  @property
  def value(self):
    # If not registered to tf.app.flags
    if self._register is None: return self._value

    assert hasattr(FLAGS, self._name)
    f_value = getattr(FLAGS, self._name)
    # Configs defined via tensorflow FLAGS have priority over any other way
    if f_value is None: return self._value
    else: return f_value

  @property
  def should_register(self):
    return self._register is not None

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

  # endregion : Class Methods

  # region : Public Methods

  def register(self, name):
    if self._name is None: self._name = name
    if self._register is None or hasattr(FLAGS, self._name): return
    self._register(self._name, None, self._description)

  def new_value(self, value):
    flg = Flag(self._default_value, self._description, self._register,
               self._name)
    flg._value = value
    return flg

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

  # region : Properties

  @property
  def should_create_path(self):
    return self.train and not self.on_cloud

  # endregion : Properties

  # region : Methods Overrode

  def __getattribute__(self, name):
    attr = object.__getattribute__(self, name)
    if not isinstance(attr, Flag): return attr
    else: return attr.value

  def __setattr__(self, name, value):
    attr = object.__getattribute__(self, name)
    if not isinstance(attr, Flag): object.__setattr__(self, name, value)
    object.__setattr__(self, name, attr.new_value(value))

  # endregion : Methods Overrode

  # region : Public Methods

  @classmethod
  def register(cls):
    queue = {name: value for name, value in cls.__dict__.items()
             if isinstance(value, Flag) and value.should_register}
    for name, flg in queue.items(): flg.register(name)

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

  # endregion : Public Methods

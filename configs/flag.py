from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tframe import tf
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
    self.name = name
    self._default_value = default_value
    self._description = description
    self._register = register
    self._is_key = is_key
    self._kwargs = kwargs

    self._value = default_value
    self._frozen = False

    # Attributes for HP searching engine
    self.hp_type = kwargs.get('hp_type', None)
    self.hp_scale = kwargs.get('hp_scale', None)
    legal_scale = ('log', 'log-uniform', 'uniform', None)
    assert self.hp_scale in legal_scale

  # region : Properties

  @property
  def ready_to_be_key(self):
    return self._is_key is None

  @property
  def is_key(self):
    if self._is_key is False: return False
    if not self.should_register: return self._is_key is True
    assert hasattr(flags.FLAGS, self.name)
    return self._is_key is True or getattr(flags.FLAGS, self.name) is not None

  @property
  def frozen(self):
    return self._frozen

  @property
  def value(self):
    # If not registered to tf.app.flags or has been frozen
    if self._register is None or self._frozen: return self._value

    assert hasattr(flags.FLAGS, self.name)
    f_value = getattr(flags.FLAGS, self.name)
    # Configs defined via tensorflow FLAGS have priority over any other way
    if f_value is None: return self._value
    # If self is en enum Flag, then f_value must be a string in
    # .. self.enum_class.value_list(), so we need to get its member
    if self.is_enum: f_value = self.enum_class.get_member(f_value)
    if self.frozen and self._value != f_value:
      raise AssertionError(
        "!! Invalid tensorflow FLAGS value {0}={1} 'cause {0} has been "
        "frozen to {2}".format(self.name, f_value, self._value))
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
  def integer(cls, default_value, description, name=None, is_key=False,
              hp_scale='uniform'):
    return Flag(default_value, description, flags.DEFINE_integer, name,
                is_key=is_key, hp_type=int, hp_scale=hp_scale)

  @classmethod
  def float(cls, default_value, description, name=None, is_key=False,
            hp_scale='uniform'):
    return Flag(default_value, description, flags.DEFINE_float, name,
                is_key=is_key, hp_type=float, hp_scale=hp_scale)

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
    if self.name is None: self.name = name
    if self._register is None or self.name in list(flags.FLAGS): return
    # Register enum flag
    if self.is_enum:
      flags.DEFINE_enum(
        self.name, None, self.enum_class.value_list(), self._description)
      return
    # Register other flag
    assert self._register is not flags.DEFINE_enum
    self._register(self.name, None, self._description)

  def new_value(self, value):
    flg = Flag(self._default_value, self._description, self._register,
               self.name, **self._kwargs)
    flg._value = value
    return flg

  def freeze(self, value):
    self._value = value
    self._frozen = True

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def parse_comma(arg, dtype=str):
    r = re.fullmatch(r'([\-\d.,]+)', arg)
    if r is None: raise AssertionError(
      'Can not parse argument `{}`'.format(arg))
    val_list = re.split(r'[,]', r.group())
    return [dtype(v) for v in val_list]

  # endregion : Private Methods


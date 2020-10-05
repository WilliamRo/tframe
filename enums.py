from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum, unique


class EnumPro(Enum):
  @classmethod
  def value_list(cls):
    return [m.value for m in list(cls.__members__.values())]

  @classmethod
  def get_member(cls, name):
    assert name in cls.value_list()
    for member in list(cls):
      if member.value == name:
        return member
    raise ValueError('!! {} not found'.format(name))


@unique
class InputTypes(EnumPro):
  BATCH = 'batch'
  RNN_BATCH = 'rnn_batch'


@unique
class SaveMode(EnumPro):
  NAIVE = 'naive'
  ON_RECORD = 'on_record'
  NOT_SAVE = 'not_save'


if __name__ == '__main__':
  assert issubclass(InputTypes, EnumPro)


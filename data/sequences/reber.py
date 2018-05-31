from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
from enum import Enum, unique

from tframe import checker
from tframe import console
from tframe.data.dataset import DataSet
from tframe.data.base_classes import DataAgent


@unique
class Symbol(Enum):
  B = 0
  T = 1
  P = 2
  S = 3
  X = 4
  V = 5
  E = 6


class ReberGrammar(object):
  # Transfer tuple
  TRANSFER = (((Symbol.T, 1), (Symbol.P, 2)),  # 0: B
              ((Symbol.S, 1), (Symbol.X, 3)),  # 1: BT
              ((Symbol.T, 2), (Symbol.V, 4)),  # 2: BP
              ((Symbol.X, 2), (Symbol.S, 5)),  # 3: BTX
              ((Symbol.P, 3), (Symbol.V, 5)),  # 4: BPV
              ((Symbol.E, None),))             # 5: BTXS or BPVV

  # Generate transfer matrix according to transfer tuple
  TRANSFER_MATRIX = np.zeros((len(TRANSFER) + 1, len(Symbol)), np.float32)
  for i, choices in enumerate(TRANSFER[:-1]):
    indices = (choices[0][0].value, choices[1][0].value)
    TRANSFER_MATRIX[i, indices] = 0.5
  TRANSFER_MATRIX[(-2, -1), (0, 6)] = 1.0

  def __init__(self, embedded=False):
    self._symbol_list = [Symbol.B]
    self.transfer_prob = None
    transfer_list = []

    # Randomly make a string
    stat = 0
    while stat is not None:
      transfer_list.append(self.TRANSFER_MATRIX[stat])
      symbol, stat = self._transfer(stat)
      self._symbol_list.append(symbol)

    # Embed
    if embedded:
      second_symbol = random.choice((Symbol.T, Symbol.P))
      self._symbol_list = ([Symbol.B, second_symbol] + self._symbol_list +
                           [second_symbol, Symbol.E])
      transfer = np.zeros((len(Symbol),), np.float32)
      transfer[second_symbol.value] = 1.0
      transfer_list = ([self.TRANSFER_MATRIX[0], self.TRANSFER_MATRIX[-2]] +
                       transfer_list + [transfer, self.TRANSFER_MATRIX[-1]])

    # Stack transfer list to form the transfer probabilities
    self.transfer_prob = np.stack(transfer_list)

  # region : Properties

  @property
  def value(self):
    return np.array([s.value for s in self._symbol_list], dtype=np.int32)

  @property
  def one_hot(self):
    result = np.zeros((len(self), len(Symbol)), np.float32)
    result[np.arange(len(self)), self.value] = 1.0
    return result[:-1]

  # endregion : Properties

  # region : Methods Overriden

  def __str__(self):
    return ''.join([s.name for s in self._symbol_list])

  def __eq__(self, other):
    return str(self) == str(other)

  def __len__(self):
    return len(self._symbol_list)

  # endregion : Methods Overriden

  # region : Public Methods

  @classmethod
  def make_strings(cls, num, unique=True, exclusive=None, embedded=False,
                   verbose=False):
    # Check input
    if exclusive is None: exclusive = []
    elif not isinstance(exclusive, list):
      raise TypeError('!! exclusive must be a list of Reber strings')
    # Make strings
    reber_list = []
    for i in range(num):
      while True:
        string = ReberGrammar(embedded)
        if unique and string in reber_list: continue
        if string in exclusive: continue
        reber_list.append(string)
        break
      if verbose:
        console.clear_line()
        console.print_progress(i + 1, num)
    if verbose: console.clear_line()
    # Return a list of Reber string
    return reber_list

  # endregion : Public Methods

  # region : Private Methods

  @classmethod
  def _transfer(cls, stat):
    return random.choice(cls.TRANSFER[stat])

  # endregion : Private Methods


class ERG(DataAgent):
  """Embedded Reber Grammar"""
  DATA_NAME = 'EmbeddedReberGrammar'

  @classmethod
  def load(cls, data_dir, train_size=256, validate_size=0, test_size=256,
           file_name=None, **kwargs):
    # Load .tfd data
    num = train_size + validate_size + test_size
    data_set = cls.load_as_tframe_data(
      data_dir, file_name=file_name, size=num, unique_=True)

    return cls._split_and_return(data_set, train_size, validate_size, test_size)


  @classmethod
  def load_as_tframe_data(cls, data_dir, file_name=None, size=512,
                          unique_=True):
    # Check file_name
    if file_name is None: file_name = cls._get_file_name(size, unique_)
    data_path = os.path.join(data_dir, file_name)
    if os.path.exists(data_path): return DataSet.load(data_path)
    # If data does not exist, create a new one
    console.show_status('Making data ...')
    erg_list = ReberGrammar.make_strings(
      size, unique_, embedded=True, verbose=True)

    # Wrap erg into a DataSet
    features = [erg.one_hot for erg in erg_list]
    targets = [erg.transfer_prob for erg in erg_list]
    data_set = DataSet(features, targets, {'erg_list': erg_list},
                       name='Embedded Reber Grammar')
    console.show_status('Saving data set ...')
    data_set.save(data_path)
    console.show_status('Data set saved to {}'.format(data_path))
    return  data_set

  @classmethod
  def _get_file_name(cls, num, unique_):
    checker.check_positive_integer(num)
    checker.check_type(unique_, bool)
    file_name = '{}_{}_{}.tfd'.format(
      cls.DATA_NAME, num, 'U' if unique_ else 'NU')
    return file_name


if __name__ == '__main__':
  console.show_status('Making data ...')
  data_set = ReberGrammar.make_strings(500, unique=True, verbose=True)
  console.show_status('{} strings have been made'.format(len(data_set)))









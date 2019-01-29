from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
from enum import Enum, unique
from collections import OrderedDict

import tensorflow as tf

from tframe import checker
from tframe import console
from tframe import context
from tframe import hub
from tframe import pedia
from tframe.nets.rnet import RNet
from tframe.data.sequences.seq_set import SequenceSet
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
  TRANSFER_MATRIX[(-2, -1), (6, 0)] = 1.0
  # Generate observation table
  OB_TABLE = np.eye(len(Symbol), dtype=np.float32)

  def __init__(self, embedded=False):
    self._symbol_list = [Symbol.B]
    self.transfer_prob = None
    self.observed_prob = None
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
      transfer_list = ([self.TRANSFER_MATRIX[0], self.TRANSFER_MATRIX[-1]] +
                       transfer_list + [transfer, self.TRANSFER_MATRIX[-2]])

    # Stack transfer list to form the transfer probabilities
    self.transfer_prob = np.stack(transfer_list)
    # Generate observation list
    self.observed_prob = np.stack(
      [self.OB_TABLE[s.value] for s in self._symbol_list[1:]])

  # region : Properties

  @property
  def value(self):
    return np.array([s.value for s in self._symbol_list], dtype=np.int32)

  @property
  def one_hot(self):
    result = np.zeros((len(self), len(Symbol)), np.float32)
    result[np.arange(len(self)), self.value] = 1.0
    return result[:-1]

  @property
  def local_binary(self):
    return np.array(self.transfer_prob > 0, dtype=self.transfer_prob.dtype)

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

  def check_grammar(self, probs):
    """Return lists of match situations for both RC and ERC criteria
       ref: AMU, 2018"""
    assert isinstance(probs, np.ndarray)
    assert len(probs) == self.value.size - 1

    def _check_token(p, q):
      assert isinstance(p, np.ndarray) and len(p.shape) == 1
      assert isinstance(q, np.ndarray) and len(q.shape) == 1
      return np.sum(p * q) == np.sum(np.sort(p) * np.sort(q))

    ERC = [_check_token(p, q) for q, p in zip(probs, self.transfer_prob)]
    return ERC[:-2], ERC

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
           file_name=None, amu18=True, cheat=True, local_binary=True,
           **kwargs):
    # Load .tfd data
    num = train_size + validate_size + test_size
    data_set = cls.load_as_tframe_data(
      data_dir, file_name=file_name, size=num, unique_=True, amu18=True,
      cheat=cheat, local_binary=local_binary)

    return cls._split_and_return(data_set, train_size, validate_size, test_size)


  @classmethod
  def load_as_tframe_data(cls, data_dir, file_name=None, size=512,
                          unique_=True, amu18=False, cheat=True,
                          local_binary=True):
    # Check file_name
    if file_name is None:
      file_name = cls._get_file_name(size, unique_, amu18, cheat, local_binary)
    data_path = os.path.join(data_dir, file_name)
    if os.path.exists(data_path): return SequenceSet.load(data_path)
    # If data does not exist, create a new one
    console.show_status('Making data ...')

    if amu18:
      train_list = ReberGrammar.make_strings(
        256, False, embedded=True, verbose=True)
      test_list = ReberGrammar.make_strings(
        256, False, embedded=True, exclusive=train_list, verbose=True)
      erg_list = train_list + test_list
    else:
      erg_list = ReberGrammar.make_strings(
        size, unique_, embedded=True, verbose=True)

    # Wrap erg into a DataSet
    features = [erg.one_hot for erg in erg_list]
    val_targets = [erg.local_binary if local_binary else erg.transfer_prob
                   for erg in erg_list]
    targets = ([erg.observed_prob for erg in erg_list]
               if not cheat else val_targets)
    # targets = [erg.transfer_prob for erg in erg_list]
    data_set = SequenceSet(
      features, targets, data_dict={'val_targets': val_targets},
      erg_list=tuple(erg_list), name='Embedded Reber Grammar')
    console.show_status('Saving data set ...')
    data_set.save(data_path)
    console.show_status('Data set saved to {}'.format(data_path))
    return data_set

  @classmethod
  def _get_file_name(cls, num, unique_, amu18, cheat, local_binary):
    checker.check_positive_integer(num)
    checker.check_type(unique_, bool)
    if amu18: tail = 'AMU18'
    elif unique_: tail = 'U'
    else: tail = 'NU'
    file_name = '{}_{}_{}_{}_{}.tfds'.format(
      cls.DATA_NAME, num, tail, 'C' if cheat else 'NC',
      'LB' if local_binary else 'P')
    return file_name

  # region : Probe Methods

  @staticmethod
  def amu18(data, trainer):
    """Probe method accepts trainer as the only parameter"""
    # region : Whatever
    acc_thres = 0.00 if hub.export_tensors_to_note else 0.8
    # Import
    import os
    from tframe.trainers.trainer import Trainer
    from tframe.models.sl.classifier import Classifier
    from tframe.data.sequences.seq_set import SequenceSet
    # Sanity check
    assert isinstance(trainer, Trainer)
    # There is no need to check RC or ERC when validation accuracy is low
    # .. otherwise a lot of time will be wasted
    if len(trainer.th.logs) == 0 or trainer.th.logs['Accuracy'] < acc_thres:
      return None
    model = trainer.model
    assert isinstance(model, Classifier)
    assert isinstance(data, SequenceSet)
    # Check state
    TERMINATED = 'TERMINATED'
    SATISFY_RC = 'SATISFY_RC'
    ERG_LIST = 'erg_list'
    if TERMINATED not in data.properties.keys():
      data.properties[TERMINATED] = False
    if data.properties[TERMINATED]: return None
    # endregion : Whatever

    probs = model.classify(data, batch_size=-1, return_probs=True)
    erg_list = data[ERG_LIST]
    RCs, ERCs = [], []
    for p, reber in zip(probs, erg_list):
      assert isinstance(reber, ReberGrammar)
      RC_detail, ERC_detail = reber.check_grammar(p)
      RCs.append(np.mean(RC_detail) == 1.0)
      ERCs.append(np.mean(ERC_detail) == 1.0)

    RC_acc, ERC_acc = np.mean(RCs), np.mean(ERCs)
    RC, ERC = RC_acc == 1, ERC_acc == 1

    counter = trainer.model.counter
    if SATISFY_RC not in data.properties.keys():
      data.properties[SATISFY_RC] = False
    if not data.properties[SATISFY_RC]:
      if RC:
        msg = ('RC is satisfied after {} sequences, '
               'test accuracy = {:.1f}%'.format(counter, 100 * ERC_acc))
        trainer.model.agent.take_notes(msg)
        data.properties[SATISFY_RC] = True
        # Take it down
        trainer.model.agent.put_down_criterion('RC', counter)
        trainer.model.agent.put_down_criterion('RC-ERC', 100 * ERC_acc)

    if ERC:
      msg = 'ERC is satisfied after {} sequences'.format(counter)
      trainer.model.agent.take_notes(msg)
      data.properties[TERMINATED] = True
      trainer.th.force_terminate = True
      # Take it down
      trainer.model.agent.put_down_criterion('ERC', counter)

    # TODO: ++export_tensors
    if hub.export_tensors_to_note:
      ERG.export_tensors(RC_acc, ERC_acc, model, data,
                         trainer.loss_history.running_average)

    msg = 'RC = {:.1f}%, ERC = {:.1f}%'.format(100 * RC_acc, 100 * ERC_acc)
    return msg

  # endregion : Probe Methods

  # region : Export tensor

  @staticmethod
  def export_tensors(RC, ERC, model, data, loss):
    agent = model.agent
    # Randomly select several samples
    num = hub.sample_num
    assert num > 0 and isinstance(num, int)
    erg_list = data.properties['erg_list']

    # Fetch tensors we need
    fetches_dict = context.tensors_to_export
    fetches = list(fetches_dict.values())
    if not hub.calculate_mean: data = data[:num]
    results = model.batch_evaluation(fetches, data)

    # Get average dy/dS for short and long trigger
    tensors = OrderedDict()
    if hub.calculate_mean: tensors['Mean'] = OrderedDict()
    exemplar_names = []
    for i in range(num):
      name = '({}){}'.format(i + 1, erg_list[i])
      tensors[name] = OrderedDict()
      exemplar_names.append(name)

    # Generate tensor dict
    for i, array_list in enumerate(results):
      name = list(fetches_dict.keys())[i]
      short_buffer = np.zeros_like(array_list[0][0][0])
      long_buffer = np.zeros_like(array_list[0][0][0])
      for j, array in enumerate(array_list):
        # The shape of array is (batch, step, *dim)
        if j < num: tensors[exemplar_names[j]][name] = array[0]
        if not hub.calculate_mean: continue
        short_buffer += np.sum(array[0], axis=0) - array[0][-2]
        long_buffer += array[0][-2]
      # Calculate mean of short/long result
      if not hub.calculate_mean: continue
      short_mean = short_buffer / (sum(data.structure) - 2 * data.size)
      long_mean = long_buffer / data.size
      tensors['Mean'][name] = np.concatenate(
        [short_mean.reshape([1, -1]), long_mean.reshape([1, -1])])

    # Take down
    scalars = OrderedDict()
    # Calculate the running average of loss
    scalars['Loss'] = loss
    scalars['RC'] = RC
    scalars['ERC'] = ERC
    agent.take_down_scalars_and_tensors(scalars, tensors)

  # endregion : Export tensor

  # region : Customized losses

  @staticmethod
  def gate_loss_for_ham(net):
    """Try to control the input_gate and output_gate of long-term units"""
    assert isinstance(net, RNet)

    # Get long-term unit gate tensors
    tensors_to_export = context.tensors_to_export
    gate_tensors = {k: v for k, v in tensors_to_export.items()
                    if k in ('out_gate_1', 'in_gate_2', 'out_gate_2')}
    long_in_gate, long_out_gate, short_out_gate = None, None, None
    for k, v in gate_tensors.items():
      assert isinstance(v, tf.Tensor)
      dim = v.shape.as_list()[-1]
      v_2d = tf.reshape(v, (-1, dim), name=k)
      if k == 'out_gate_1': short_out_gate = v_2d
      elif k == 'in_gate_2': long_in_gate = v_2d
      elif k == 'out_gate_2': long_out_gate = v_2d

    loss_tensors = []
    coef = hub.gate_loss_strength
    def get_loss(name, gate, index, suppress=False):
      assert isinstance(gate, tf.Tensor)
      name = '{}_loss'.format(name)
      dim = gate.shape.as_list()[-1]
      if hub.apply_default_gate_loss:
        loss = tf.reduce_sum(gate)
      elif suppress:
        loss = tf.reduce_sum(gate) + dim - 2 * tf.reduce_sum(gate[index])
      else:
        loss = tf.reduce_sum(gate[index])
      console.show_status('{} `{}` added.'.format(
        'Default' if hub.apply_default_gate_loss else 'Customized', name))
      return tf.multiply(loss, coef, name=name)

    if long_in_gate is not None:
      loss_tensors.append(get_loss('long_in_gate', long_in_gate, 1))
    if long_out_gate is not None:
      loss_tensors.append(get_loss('long_out_gate', long_out_gate, -2))
    if short_out_gate is not None:
      loss_tensors.append(get_loss('short_out_gate', short_out_gate, -2,
                                   suppress=True))
    return loss_tensors

  # endregion : Customized losses


if __name__ == '__main__':
  console.show_status('Making data ...')
  data_set = ReberGrammar.make_strings(
    5, unique=True, verbose=True, embedded=True)
  console.show_status('{} strings have been made'.format(len(data_set)))









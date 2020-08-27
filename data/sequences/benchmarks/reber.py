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
  # TM[-2]: E; TM[-1]: B
  TRANSFER_MATRIX[(-2, -1), (-1, 0)] = 1.0
  # Generate observation table
  OB_TABLE = np.eye(len(Symbol), dtype=np.float32)

  def __init__(self, embedded=False, multiple=1, specification=None):
    assert specification in (None, 'T', 'P')
    self._symbol_list = [Symbol.B]
    self._sub_rebers = [[Symbol.B]]
    self._multiple = checker.check_positive_integer(multiple)
    self.transfer_prob = None
    self.observed_prob = None
    transfer_list = []

    # BT(BTXSE)(BPVVE)TE  <- stat
    # TB TSXEB  TTPET E   <- transfer
    # P  PXS    PVV

    # Randomly make a string
    stat, count = 0, 0
    while stat is not None:
      transfer_list.append(self.TRANSFER_MATRIX[stat])
      symbol, stat = self._transfer(stat)
      self._symbol_list.append(symbol)
      self._sub_rebers[count].append(symbol)
      # Generate next embedded if necessary
      if stat is None:
        count += 1
        if count < self._multiple:
          self._sub_rebers.append([])
          self._symbol_list.append(Symbol.B)
          self._sub_rebers[count].append(symbol.B)
          transfer_list.append(self.TRANSFER_MATRIX[-1])
          stat = 0
    assert len(self._sub_rebers) == self._multiple

    # Embed
    if embedded:
      if specification == 'T': second_symbol = Symbol.T
      elif specification == 'P': second_symbol = Symbol.P
      else: second_symbol = random.choice((Symbol.T, Symbol.P))

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
  def lc_indices(self):
    indices = [i for i, s in enumerate(self._symbol_list) if s is Symbol.E]
    return indices[:-1]

  @property
  def sc_indices(self):
    indices = set(range(len(self) - 1)) - set(self.lc_indices)
    return list(indices)

  @property
  def abbreviation(self):
    reber = str(self)
    max_len = 25
    expand = len(reber) < max_len
    result = reber[:2]
    for sub in self._sub_rebers:
      assert isinstance(sub, list)
      sub_str = '({})'.format(
        ''.join([s.name for s in sub]) if expand else 'L:{}'.format(len(sub)))
      result += sub_str
    return result + reber[-2:]

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
                   multiple=1, verbose=False, interleave=True):
    # Check input
    if exclusive is None: exclusive = []
    elif not isinstance(exclusive, list):
      raise TypeError('!! exclusive must be a list of Reber strings')
    # Make strings
    reber_list = []
    long_token = None
    for i in range(num):
      if interleave: long_token = 'T' if long_token in ('P', None) else 'P'
      while True:
        string = ReberGrammar(
          embedded, multiple=multiple, specification=long_token)
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

  def check_grammar_amu18(self, probs):
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

  def check_grammar_gdu19(self, probs):
    """Check short and long criteria"""
    assert isinstance(probs, np.ndarray)
    assert len(probs) == self.value.size - 1

    def _check_token(p, q):
      assert isinstance(p, np.ndarray) and len(p.shape) == 1
      assert isinstance(q, np.ndarray) and len(q.shape) == 1
      return np.sum(p * q) == np.sum(np.sort(p) * np.sort(q))

    ACC = [_check_token(p, q) for q, p in zip(probs, self.transfer_prob)]
    # SC, LC, ALL
    return ACC[:-2] + [ACC[-1]], [ACC[-2]], ACC

  def check_grammar(self, probs):
    """Check short and long criteria"""
    assert isinstance(probs, np.ndarray)
    assert len(probs) == self.value.size - 1

    def _check_token(p, q):
      assert isinstance(p, np.ndarray) and len(p.shape) == 1
      assert isinstance(q, np.ndarray) and len(q.shape) == 1
      return np.sum(p * q) == np.sum(np.sort(p) * np.sort(q))

    ACC = [_check_token(p, q) for q, p in zip(probs, self.transfer_prob)]
    ACC = np.array(ACC)
    # SC, LC, ALL
    return ACC[self.sc_indices], ACC[self.lc_indices], ACC

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
  def load(cls, data_dir, train_size=1000, validate_size=0, test_size=200,
           file_name=None, cheat=True, local_binary=False, multiple=1,
           rule=None, **kwargs):
    # Load .tfd data
    num = train_size + validate_size + test_size
    data_set = cls.load_as_tframe_data(
      data_dir, file_name=file_name, train_size=train_size, test_size=test_size,
      unique_=True, cheat=cheat, local_binary=local_binary, multiple=multiple,
      rule=rule)

    return cls._split_and_return(data_set, train_size, validate_size, test_size)


  @classmethod
  def load_as_tframe_data(cls, data_dir, train_size=1000, test_size=200,
                          file_name=None, unique_=True, cheat=True,
                          local_binary=True, multiple=1, rule=None):
    assert rule in ('lstm97', 'pau19', None)

    # Check file_name
    if file_name is None:
      file_name = cls._get_file_name(
        train_size, test_size, unique_, cheat, local_binary, multiple, rule)
    data_path = os.path.join(data_dir, file_name)
    if os.path.exists(data_path): return SequenceSet.load(data_path)
    # If data does not exist, create a new one
    console.show_status('Making data ...')

    if rule == 'pau19':
      erg_list = ReberGrammar.make_strings(
        train_size + test_size, True, embedded=True, multiple=multiple,
        verbose=True)
    elif rule == 'lstm97':
      train_list = ReberGrammar.make_strings(
        train_size, False, embedded=True, verbose=True, multiple=multiple)
      test_list = ReberGrammar.make_strings(
        test_size, False, embedded=True, exclusive=train_list, verbose=True,
        multiple=multiple)
      erg_list = train_list + test_list
    else:
      erg_list = ReberGrammar.make_strings(
        train_size + test_size, unique_, embedded=True, verbose=True,
        multiple=multiple)

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
  def _get_file_name(cls, train_size, test_size, unique_, cheat, local_binary,
                     multiple, rule):
    checker.check_positive_integer(train_size)
    checker.check_positive_integer(test_size)
    checker.check_positive_integer(multiple)
    checker.check_type(unique_, bool)
    if rule is not None: tail = rule
    elif unique_: tail = 'U'
    else: tail = 'NU'
    file_name = '{}{}_{}+{}_{}_{}_{}.tfds'.format(
      cls.DATA_NAME, '' if multiple == 1 else '(x{})'.format(multiple),
      train_size, test_size, tail, 'C' if cheat else 'NC',
      'LB' if local_binary else 'P')
    if multiple > 1: file_name = 'm' + file_name
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
      RC_detail, ERC_detail = reber.check_grammar_amu18(p)
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
                         trainer.batch_loss_stat.running_average)

    msg = 'RC = {:.1f}%, ERC = {:.1f}%'.format(100 * RC_acc, 100 * ERC_acc)
    return msg


  @staticmethod
  def probe(data, trainer):
    """New probe method using long/short criteria"""

    # region : Preparation
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
    # if len(trainer.th.logs) == 0 or trainer.th.logs['Accuracy'] < acc_thres:
    #   return None # TODO
    model = trainer.model
    agent = model.agent
    assert isinstance(model, Classifier)
    assert isinstance(data, SequenceSet)
    # Check state
    SATISFY_SHORT = 'SATISFY_SHORT'
    SATISFY_LONG = 'SATISFY_LONG'
    ERG_LIST = 'erg_list'
    # endregion : Preparation

    # region : Calculate 2 criteria

    probs = model.classify(data, batch_size=-1, return_probs=True)
    erg_list = data[ERG_LIST]
    SHORTs, LONGs, ALLs = [], [], []
    for p, reber in zip(probs, erg_list):
      assert isinstance(reber, ReberGrammar)
      SHORT_detail, LONG_detail, ALL_detail = reber.check_grammar(p)
      SHORTs.append(np.mean(SHORT_detail) == 1.0)
      LONGs.append(np.mean(LONG_detail) == 1.0)
      ALLs.append(np.mean(ALL_detail) == 1.0)

    SHORT_acc, LONG_acc, ALL_acc = np.mean(SHORTs), np.mean(LONGs), np.mean(ALLs)
    SHORT, LONG, ALL = SHORT_acc == 1, LONG_acc == 1, ALL_acc == 1

    # endregion : Calculate 2 criteria

    # region : Check 2 criteria

    counter = model.counter
    for key, criterion, accuracy in zip(
        (SATISFY_SHORT, SATISFY_LONG), (SHORT, LONG), (SHORT_acc, LONG_acc)):
      if key not in data.properties.keys(): data.properties[key] = False
      if not data.properties[key] and criterion:
        if key == SATISFY_SHORT: name, op_name, op_acc = 'SC', 'LC', LONG_acc
        else: name, op_name, op_acc = 'LC', 'SC', SHORT_acc
        # Write msg to note
        msg = (
          '{} is satisfied after {} sequences, {} accuracy = {:.2f}%'.format(
            name, counter, op_name, 100 * op_acc))
        agent.take_notes(msg)
        data.properties[key] = True
        # Take it down to note for view
        agent.put_down_criterion(name, counter)
        # agent.put_down_criterion('{}({}S)'.format(op_name, name), 100 * op_acc)

    if ALL:
      trainer.th.force_terminate = True
      agent.take_notes('ALL is satisfied after {} sequences.'.format(counter))
      agent.put_down_criterion('ALL', counter)

    # endregion : Check 2 criteria

    # region : Export and return

    if hub.export_tensors_to_note:
      ERG.export_tensors(
        SHORT_acc, LONG_acc, trainer, data,
        trainer.batch_loss_stat.running_average, 'SC', 'LC')

    return 'S = {:.1f}%, L = {:.1f}%, A = {:.1f}%'.format(
      100 * SHORT_acc, 100 * LONG_acc, 100 * ALL_acc)

    # endregion : Export and return

  # endregion : Probe Methods

  # region : Export tensor

  @staticmethod
  def export_tensors(
      CR1, CR2, trainer, data, loss, CR1_STR='RC', CR2_STR='ERC'):
    """TODO: this method can be replaced by the build-in methods in
             trainer.py
    """
    model = trainer.model
    agent = model.agent
    # Randomly select several samples
    num = hub.sample_num
    assert num > 0 and isinstance(num, int)
    erg_list = data.properties['erg_list']

    # Fetch tensors we need
    fetches_dict = context.tensors_to_export
    fetches = list(fetches_dict.values())
    if not hub.calculate_mean: data = data[:num]
    results = model.evaluate(fetches, data)

    # Get average dy/dS for short and long trigger
    tensors = OrderedDict()
    if hub.calculate_mean: tensors['Mean'] = OrderedDict()
    exemplar_names = []
    for i in range(num):
      r = erg_list[i]
      assert isinstance(r, ReberGrammar)
      name = '({}){}'.format(i + 1, r.abbreviation)
      tensors[name] = OrderedDict()
      exemplar_names.append(name)

    # Generate tensor dict
    for i, array_list in enumerate(results):
      name = list(fetches_dict.keys())[i]
      # short_buffer = np.zeros_like(array_list[0][0][0])
      # long_buffer = np.zeros_like(array_list[0][0][0])
      for j, array in enumerate(array_list):
        # The shape of array is (batch, step, *dim)
        # why array[0] previously?
        # if j < num: tensors[exemplar_names[j]][name] = array[0]
        if j < num: tensors[exemplar_names[j]][name] = array
        if not hub.calculate_mean: continue
        # short_buffer += np.sum(array[0], axis=0) - array[0][-2]
        # long_buffer += array[0][-2]
      # Calculate mean of short/long result
      if not hub.calculate_mean: continue
      # short_mean = short_buffer / (sum(data.structure) - 2 * data.size)
      # long_mean = long_buffer / data.size
      # tensors['Mean'][name] = np.concatenate(
      #   [short_mean.reshape([1, -1]), long_mean.reshape([1, -1])])

    if len(results) == 0: tensors = OrderedDict()
    # Add variables to export
    tensors = trainer.get_variables_to_export(tensors)

    # Take down
    scalars = OrderedDict()
    # Calculate the running average of loss
    scalars['Loss'] = loss
    scalars[CR1_STR] = CR1
    scalars[CR2_STR] = CR2
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
    5, unique=True, verbose=True, embedded=True, multiple=1)
  console.show_status('{} strings have been made'.format(len(data_set)))









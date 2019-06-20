from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import checker

from tframe.data.dataset import DataSet
from tframe.data.sequences.paral_engine import ParallelEngine


class SequenceSet(DataSet):
  """A SequenceSet stores lists of sequences in data_dict. Each sequence must
     be a numpy array and its length may vary. Each sequence list shares
     a same `structure` defined as [len(s) for s in sequence_list], which
     is a property of the SequenceSet class.

     In seq2seq tasks, the targets should be stored in data_dict so that
     target list will be forced to share structure with feature list.
     While in sequence classification tasks, `n_to_one` attribute should be
     set to True and targets will be stored in `summ_dict` in which each
     list shares a same length.
  """

  EXTENSION = 'tfds'

  PARALLEL_ON = 'PARALLEL_ON'
  DATA_STACK = 'DATA_STACK'
  PADDED_STACK = 'PADDED_STACK'

  def __init__(self, features=None, targets=None, data_dict=None,
               summ_dict=None, n_to_one=False, name='seqset', **kwargs):
    """
    A SequenceSet stores lists of sequences.
    :param summ_dict: stores summaries of sequences.
    """
    # Attributes
    self.summ_dict = {} if summ_dict is None else summ_dict
    kwargs['n_to_one'] = checker.check_type(n_to_one, bool)
    if n_to_one and targets is not None:
      self.summ_dict['targets'] = targets
      targets = None

    # Call parent's constructor
    super().__init__(features, targets, data_dict, name, **kwargs)


  # region : Properties

  @DataSet.features.setter
  def features(self, val):
    if val is not None:
      if isinstance(val, list): self.data_dict[self.FEATURES] = val
      else: raise TypeError('!! Unsupported feature type {}'.format(type(val)))

  @DataSet.targets.setter
  def targets(self, val):
    if val is not None:
      if isinstance(val, list): self.data_dict[self.TARGETS] = val
      else: raise TypeError('!! Unsupported target type {}'.format(type(val)))

  @property
  def representative(self):
    sequences = list(self.data_dict.values())[0]
    assert isinstance(sequences, list)
    return sequences

  @property
  def structure(self): return [len(s) for s in self.representative]

  @property
  def is_regular_array(self): return False

  @property
  def equal_length(self):
    s = self.structure
    return all(np.array(s) == s[0])

  @property
  def merged_data_dict(self):
    """Merge summ_dict to data_dict by duplicating summaries"""
    merged_dict = self.data_dict.copy()
    for name, summ_list in self.summ_dict.items():
      full_data = []
      for summ, seq_len in zip(summ_list, self.structure):
        if np.isscalar(summ): summ = np.array([summ]).reshape(1, 1)
        assert isinstance(summ, np.ndarray) and summ.shape[0] == 1
        if len(summ.shape) == 1: summ = np.reshape(summ, (1, 1))
        full_data.append(np.broadcast_to(summ, (seq_len, *summ.shape[1:])))
      merged_dict[name] = full_data
    return merged_dict

  @property
  def stack(self):
    """Concatenate this sequence set (a list consists of sequences with shape
       [steps, *dim]) to a regular array with shape [sum(steps), *dim]"""
    if self.DATA_STACK in self.properties.keys():
      stack = self.properties[self.DATA_STACK]
      assert isinstance(stack, DataSet)
      return stack
    self.properties[self.DATA_STACK] = DataSet(
      data_dict=self._apply(np.concatenate, self.merged_data_dict),
      name=self.name + '(stacked)', **self.properties)
    return self.stack
  
  @property
  def padded_stack(self):
    """Stack this sequence set with 0 padded. The output shape is
       (self.size, max_steps, *dim)"""
    if self.PADDED_STACK in self.properties.keys():
      stack = self.properties[self.PADDED_STACK]
      assert isinstance(stack, DataSet)
      return stack
    max_step = max(self.structure)
    f = lambda seqs: self._pad_sequences(seqs, max_step)
    self.properties[self.PADDED_STACK] = DataSet(
      data_dict=self._apply(f, self.merged_data_dict),
      name=self.name + '(padded_stack)', is_rnn_input=True, **self.properties)
    self.padded_stack.active_length = self.structure
    return self.padded_stack

  @property
  def as_rnn_batch(self):
    """Used in one-shot validation"""
    return self.padded_stack

  @property
  def parallel_on(self):
    return self.properties.get(self.PARALLEL_ON, False)

  # endregion : Properties

  # region : Overriden Methods

  def __getitem__(self, item):
    if isinstance(item, str):
      if item in self.data_dict.keys(): return self.data_dict[item]
      elif item in self.summ_dict.keys(): return self.summ_dict[item]
      elif item in self.properties.keys(): return self.properties[item]
      else: raise KeyError('!! Can not resolve "{}"'.format(item))

    # If item is index array
    f = lambda x: self._get_subset(x, item)
    data_set = SequenceSet(
      data_dict=self._apply(f), summ_dict=self._apply(f, self.summ_dict),
      name=self.name + '(slice)')
    return self._finalize(data_set, item)

  # endregion : Overriden Methods

  # region : Basic APIs

  def get_round_length(self, batch_size, num_steps=None):
    if num_steps is None:
      # when num_steps is None, this seq_set is treated as common data_set
      # and always contains single sequence
      if self.batch_preprocessor is None:
        return self.stack.get_round_length(batch_size)
      else:
        if self.length_calculator is None:
          # TODO: restrict batch preprocessors not to change sequence length
          return self.stack.get_round_length(batch_size)
          # return None
        else:
          L = sum([self.length_calculator(l) for l in self.structure])
          round_len = np.ceil(L / batch_size)
    else:  # else generate RNN batches
      if self.parallel_on:
        return self._get_pe_round_length(batch_size, num_steps)
      else:
        assert isinstance(batch_size, int)
        if batch_size < 0: batch_size = self.size

        if batch_size == 1:
          round_len = sum([np.ceil(L / (num_steps if num_steps > 0 else L))
                           for L in self.structure])
        else:
          assert num_steps < 0
          round_len = np.ceil(self.size / batch_size)
    return int(round_len)

  def gen_rnn_batches(self, batch_size=1, num_steps=-1, shuffle=False,
                      is_training=False):
    """Generate RNN batches in which each data item has shape
      (batch_size, steps, *sample_shape)
      If parallel option is on, batches will be yielded from a BETA method
      Otherwise for training, batch_size should be set to 1.

      (batch_size, num_steps) combination can be:
      (1) batch_size = 1

      (2) batch_size > 1, if batch_size is -1, it will be set to self.size
          (a) num_steps is forced to be -1,
             i.e. sequence partition is forbidden.

       Here batch pre-processor is not considered

    :param batch_size: integer. When is not training, this value can be set
                        to -1 or any positive integer.
    :param num_steps:  a non-negative integer.
    :param shuffle:    Whether to shuffle
    """
    # BETA, only available for model training
    if self.parallel_on:
      yield from self._gen_parallel_batches(batch_size, num_steps, shuffle)
      return

    # Get round length
    round_len = self.get_round_length(batch_size, num_steps)
    # If batch_size < 0, set it to self.size
    if batch_size < 0: batch_size = self.size
    # Calculate # batches in one epoch
    num_batches = int(np.ceil(self.size / batch_size))
    # counter helps to enforce the total iterations in one epoch matches the
    #  round length
    counter = 0
    # Initialize indices, shuffle if necessary
    self._init_indices(shuffle)
    for i in range(num_batches):
      # Get sequence list of length `batch_size`
      indices = self._select(i, batch_size, shuffle)
      seq_batch = self[indices]
      active_length = None
      # Pre-proceed this batch if necessary
      # preprocessor should be used very carefully
      if self.batch_preprocessor is not None:
        seq_batch = self.batch_preprocessor(seq_batch, is_training)
        seq_batch.remove_batch_preprocessor()

      if isinstance(seq_batch, SequenceSet):
        # If batch_size > 1, calculate active_length in case sequences in this
        #   batch have various lengths
        if seq_batch.size > 1:
          active_length = seq_batch.structure
          # forbid sequence partition
          assert num_steps < 0
        # padded_stack is_rnn_input, and is a DataSet, not SeqSet any more
        seq_batch = seq_batch.padded_stack
        # At this point, seq_batch is a DataSet with active_length being
        #   not None

      # seq_batch.shape = (batches, steps, *shape)
      # Use DataSet's gen_rnn_batches method to yield batches
      # Note that here `batch_size` parameter will not be used in
      #  seq_batch.gen_rnn_batches method
      for batch in seq_batch.gen_rnn_batches(
          1, num_steps, is_training=is_training):
        batch.active_length = active_length
        yield batch
        counter += 1
        if counter == round_len: break

      if counter == round_len: break

    if not shuffle: assert counter == round_len

  # endregion : Basic APIs

  # region : Public Methods

  def turn_parallel_on(self): self.properties[self.PARALLEL_ON] = True

  # endregion : Public Methods

  # region : Private Methods

  def _finalize(self, data_set, indices=None):
    data_set = super()._finalize(data_set, indices)
    data_set.properties.pop(self.DATA_STACK, None)
    data_set.properties.pop(self.PADDED_STACK, None)
    if self.num_classes is not None and 'targets' in self.summ_dict.keys():
      data_set.refresh_groups()
    return data_set

  def _check_data(self):
    """data_dict should be a non-empty dictionary containing equilength lists of
       regular numpy arrays. Samples in the same sequence list must have the
       same shape
       summ_dict should be a dictionary which stores summaries of each sequence.
   """
    # Check data_dict and summ_dict
    if not isinstance(self.data_dict, dict) or len(self.data_dict) == 0:
      raise TypeError('!! data_dict must be a non-empty dictionary')
    if not isinstance(self.summ_dict, dict):
      raise TypeError('!! summ_dict must be a dictionary')

    list_length = len(list(self.data_dict.values())[0])

    # Check each item in data_dict
    for name, seq_list in self.data_dict.items():
      checker.check_type(seq_list, np.ndarray)
      # Check type and length
      if not isinstance(seq_list, list) or len(seq_list) != list_length:
        raise ValueError('!! {} should be a list with length {}'.format(
          name, list_length))
      # Check structure
      if [len(s) for s in seq_list] != self.structure:
        raise ValueError('!! sequence list structure inconformity: {} '.format(
          name))
      # Make sure len(sample_shape) > 0
      if len(seq_list[0].shape) < 2:
        seq_list = [s.reshape(-1, 1) for s in seq_list]
      # Check sample shape
      shapes = [s.shape[1:] for s in seq_list]
      if shapes.count(shapes[0]) != len(shapes):
        raise AssertionError('!! Sample shape in {} are inconformity'.format(
          name))

      self.data_dict[name] = seq_list

    # Check each item in summ_dict
    for name, summ_list in self.summ_dict.items():
      # Check type and length
      if not isinstance(summ_list, list) or len(summ_list) != list_length:
        raise ValueError('!! {} should be a list of length {}'.format(
          name, list_length))

      if checker.check_scalar_list(summ_list): continue

      checker.check_type(summ_list, np.ndarray)
      # Check structure
      for i, summ in enumerate(summ_list):
        if summ.shape[0] > 1: summ_list[i] = np.reshape(summ, (1, *summ.shape))

      # Check sample shape
      shapes = [s.shape[1:] for s in summ_list]
      if shapes.count(shapes[0]) != len(shapes):
        raise AssertionError('!! Sample shape in {} are inconformity'.format(
          name))

  @staticmethod
  def _pad_sequences(sequences, max_steps):
    """Receive a list of irregular sequences and output a regular numpy array"""
    assert isinstance(sequences, list)
    checker.check_positive_integer(max_steps)
    checker.check_type(sequences, np.ndarray)

    if all([s.shape[0] == sequences[0].shape[0] for s in sequences]):
      return np.stack(sequences, axis=0)

    sample_shape = sequences[0].shape[1:]
    assert len(sample_shape) > 0
    stack = np.zeros(shape=(len(sequences), max_steps, *sample_shape))
    for i, s in enumerate(sequences): stack[i, :len(s)] = s

    return stack

  # endregion : Private Methods

  # region : BETA

  def _get_pe_round_length(self, batch_size, num_steps):
    if (self.batch_preprocessor is not None
        and self.length_calculator is None):
      return 10000 # TODO: default round length
    if self.batch_preprocessor is None: assert self.length_calculator is None

    return ParallelEngine.get_round_length(
      batch_size, num_steps, self.structure, len_f=self.length_calculator)

  def _gen_parallel_batches(self, batch_size, num_steps, shuffle):
    """A beta method used only for RNN training. Both features and targets
       are required"""
    # Sanity check
    if not self.parallel_on: raise ValueError(
      '!! parallel engine is not activated')
    assert isinstance(batch_size, int) and isinstance(num_steps, int)
    assert isinstance(shuffle, bool)
    if batch_size < 0: batch_size = self.size

    # Initialize parallel engine
    pe = ParallelEngine(batch_size)
    round_len = self._get_pe_round_length(batch_size, num_steps)

    # Start loop
    global_reset = True
    counter, cursor = 0, 0
    while True:
      reset_indices = pe.inactive_indices
      reset_values = []

      # Load new sequence to engine if necessary
      while not pe.is_ready:
        if shuffle or cursor < self.size:
          index = self._rand_indices() if shuffle else cursor
          ds = self[index].stack
          if self.batch_preprocessor is not None:
            ds = self.batch_preprocessor(ds)
          cursor += 1
          reset_values.append(0)
        else:
          ds = None
          reset_values.append(None)
        pe.set_data(ds)

      if pe.flameout: break

      # Get data batch
      data_batch = pe.emit(num_steps)
      if len(reset_indices) > 0:
        if global_reset:
          data_batch.should_reset_state = True
          global_reset = False
        assert len(reset_indices) == len(reset_values)
        data_batch.reset_batch_indices = reset_indices
        data_batch.reset_values = (
          reset_values if len([val for val in reset_values if val is None]) > 0
          else None)

      # Yield batch
      yield  data_batch

      counter += 1
      if counter >= round_len: break

    # Check round length
    assert counter == round_len

  # endregion : BETA





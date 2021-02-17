from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import checker
from tframe import pedia
from tframe import hub

from tframe.utils import misc

from tframe.data.base_classes import TFRData


class DataSet(TFRData):

  EXTENSION = 'tfd'

  FEATURES = pedia.features
  TARGETS = pedia.targets

  def __init__(self, features=None, targets=None, data_dict=None,
               name='dataset', is_rnn_input=False, **kwargs):
    """
    A DataSet is the only data structure which can be fed into tframe model
    directly. Data stored in data_dict must be a regular numpy array with the
    same length.
    """
    # Call parent's constructor
    super().__init__(name)

    # Attributes
    self.data_dict = {} if data_dict is None else data_dict
    self.features = features
    self.targets = targets
    self.properties.update(kwargs)

    self.is_rnn_input = is_rnn_input
    self.active_length = None
    self.should_reset_state = False
    self.reset_batch_indices = None
    self.reset_values = None
    self.active_indices = None

    # Sanity checks
    self._check_data()

    # Indices
    self.indices = None
    self._ordered_indices = np.array(list(range(self.size)))

  # region : Properties

  @property
  def gather_indices(self):
    assert isinstance(self.active_length, (list, tuple))
    return [[i, al - 1] for i, al in enumerate(self.active_length)]
  
  @property
  def features(self): return self.data_dict.get(self.FEATURES, None)

  @features.setter
  def features(self, val):
    if val is not None:
      if isinstance(val, np.ndarray): self.data_dict[self.FEATURES] = val
      else: raise TypeError('!! Unsupported feature type {}'.format(type(val)))

  @property
  def targets(self): return self.data_dict.get(self.TARGETS, None)

  @targets.setter
  def targets(self, val):
    if val is not None:
      if isinstance(val, np.ndarray): self.data_dict[self.TARGETS] = val
      else: raise TypeError('!! Unsupported target type {}'.format(type(val)))

  @property
  def dense_labels(self):
    if self.DENSE_LABELS in self.data_dict:
      return self.data_dict[self.DENSE_LABELS]
    if self.num_classes is None: raise AssertionError(
      '!! # classes should be known for getting dense labels')
    # Try to convert dense labels from targets
    targets = self.targets
    # Handle sequence summary situation
    if isinstance(targets, (list, tuple)):
      targets = np.concatenate(targets, axis=0)
    dense_labels = misc.convert_to_dense_labels(targets)
    self.dense_labels = dense_labels
    return dense_labels

  @dense_labels.setter
  def dense_labels(self, val):
    self.data_dict[self.DENSE_LABELS] = val

  @property
  def n_to_one(self):
    return self.properties.get('n_to_one', False)

  @property
  def representative(self):
    array = list(self.data_dict.values())[0]
    assert isinstance(array, np.ndarray)
    assert len(array.shape) > 2 if self.is_rnn_input else 1
    return array

  @property
  def should_partially_reset_state(self):
    return self.reset_batch_indices is not None

  @property
  def structure(self): return [1]

  @property
  def size(self): return len(self.representative)

  @property
  def total_steps(self):
    assert self.is_rnn_input
    return self.representative.shape[1]

  @property
  def is_regular_array(self): return True

  @property
  def stack(self): return self

  @property
  def as_rnn_batch(self):
    """Convert a regular array to RNN batch format"""
    if self.is_rnn_input: return self
    return self._convert_to_rnn_input(training=False)

  @property
  def feature_mean(self):
    from tframe.data.sequences.seq_set import SequenceSet
    assert not isinstance(self, SequenceSet)
    return np.mean(self.features, axis=0)

  @property
  def feature_std(self):
    from tframe.data.sequences.seq_set import SequenceSet
    assert not isinstance(self, SequenceSet)
    return np.std(self.features, axis=0)

  # endregion : Properties

  # region : Overriden Methods

  def __len__(self): return self.size

  def __getitem__(self, item):
    if isinstance(item, str):
      if item in self.data_dict.keys(): return self.data_dict[item]
      elif item in self.properties.keys(): return self.properties[item]
      else: raise KeyError('!! Can not resolve "{}"'.format(item))

    # If item is index array
    f = lambda x: self._get_subset(x, item)

    data_set = type(self)(data_dict=self._apply(f), name=self.name + '(slice)')
    return self._finalize(data_set, item)

  # endregion : Overriden Methods

  # region : Basic APIs

  def get_round_length(self, batch_size, num_steps=None, training=False):
    assert isinstance(batch_size, int) and isinstance(training, bool)
    if batch_size < 0: batch_size = self.size

    if num_steps is None: round_len = np.ceil(self.size / batch_size)
    else:
      if self.is_rnn_input:
        if num_steps < 0: num_steps = self.total_steps
        round_len = np.ceil(self.total_steps / num_steps)
      else:
        if num_steps < 0: round_len = 1
        elif training and hub.random_sample_length is not None:
          # This branch is under testing
          L = checker.check_positive_integer(hub.random_sample_length)
          round_len = int(np.ceil(L / num_steps))
        else:
          # e.g. PTB
          M, N, p = self.size, batch_size, hub.overlap_pct if training else 0
          assert 0 <= p < 1
          L = int(M/((N - 1)*(1 - p) + 1))
          round_len = int(np.ceil(L / num_steps))

    round_len = int(round_len)
    if training: self._set_dynamic_round_len(round_len)
    return round_len

  def gen_batches(self, batch_size, shuffle=False, is_training=False):
    """Yield batches of data with the specific size"""
    round_len = self.get_round_length(batch_size, training=is_training)
    if batch_size == -1: batch_size = self.size

    # Generate batches
    self._init_indices(shuffle)
    for i in range(round_len):
      indices = self._select(i, batch_size, training=is_training)
      # Get subset
      data_batch = self[indices]
      # Preprocess if necessary
      if self.batch_preprocessor is not None:
        data_batch = self.batch_preprocessor(data_batch, is_training)
      # Make sure data_batch is a regular array
      if not data_batch.is_regular_array: data_batch = data_batch.stack
      # Yield data batch
      yield data_batch

    # Clear dynamic_round_len if necessary
    if is_training: self._clear_dynamic_round_len()

  def gen_rnn_batches(self, batch_size=1, num_steps=-1, shuffle=False,
                      is_training=False, act_lens=None):
    """Each numpy array will be regarded as a single sequence and will be
       partitioned into batches of sequences with corresponding steps.

      :param batch_size: Batch size, positive integer
                         used for converting self to RNN input,
                         if self is already an RNN input, this parameter
                         will not be used
      :param num_steps: steps of each RNN data batch
      :param shuffle: This parameter must be False here
      :param is_training: Whether this method is invoked by tframe Trainer
                          during training
      :param act_lens: Must be provided when invoked by
                       SequenceSet.gen_rnn_batches
     """
    # Check batch_size and shuffle
    checker.check_positive_integer(batch_size)
    assert shuffle is False
    # Generate partitioned data set
    if self.is_rnn_input: rnn_data = self
    else:
      data_set = (self if self.batch_preprocessor is None
                  else self.batch_preprocessor(self, is_training))
      # Total steps will be data_size // batch_size, i.e. data may be
      # .. truncated
      # overlap_pct and random_shift takes effect here
      rnn_data = data_set._convert_to_rnn_input(is_training, batch_size)

    # here each entry in data_dict has shape [batch_size, steps, *dim]
    if act_lens is None:
      if num_steps < 0 or num_steps is None: num_steps = rnn_data.total_steps
      round_len = self.get_round_length(batch_size, num_steps, is_training)
    else:
      # This branch will be visited only when this method is called by
      # .. SequenceSet.gen_rnn_batches. Thus set_dynamic_round_len is not
      # .. necessary
      assert isinstance(act_lens, list) and len(act_lens) > 0
      if num_steps < 0:
        num_steps = min(act_lens) if is_training else max(act_lens)
      round_len = self._get_dynamic_round_len(act_lens, num_steps, is_training)

    # Generate batches
    def extract(i, f, data=None):
      assert callable(f) and isinstance(i, int) and i >= 0
      if data is None: data = rnn_data
      batch = DataSet(
        data_dict=data._apply(f), is_rnn_input=True,
        name=self.name + '_batch_{}_of_{}'.format(i + 1, round_len),
        **self.properties)
      # Signal for predictor's _get_default_feed_dict method
      if i == 0: batch.should_reset_state = True
      return batch
    i = 0
    if act_lens is None:
      for i in range(round_len):
        # Last data_batch may be shorter. (a = [1, 2], a[:9] is legal)
        f = lambda x: x[:, i*num_steps:(i + 1)*num_steps]
        # Yield data batch
        yield extract(i, f)
    else:
      training = is_training
      # This block is the mirror of _get_dynamic_round_len
      # TODO: should be refactored to be more elegant
      assert isinstance(act_lens, list) and len(act_lens) > 0
      checker.check_positive_integer(num_steps)
      i, start, decrease, indices = 0, 0, False, None
      while len(act_lens) > 0:
        assert rnn_data.size == len(act_lens)
        # Decide steps to sample
        sl = min(act_lens)
        assert sl > 0
        L = min(sl, num_steps) if training else num_steps
        # Extract
        f = lambda x: x[:, start:(start + L)]
        batch = extract(i, f, data=rnn_data)
        # Set active_length to batch
        batch.active_length = [L if training else min(al, L) for al in act_lens]

        # Set decrease signal to decrease batch size
        if decrease:
          batch.active_indices = indices
          decrease = False
        # Yield data_batch
        yield batch

        # Calculate new active_length
        new_lens = [al - L for al in act_lens]
        indices = [i for i, al in enumerate(new_lens) if al > 0]
        # Update rnn_data if any sequence has been finished
        if 0 < len(indices) < len(act_lens):
          decrease = True
          rnn_data = rnn_data[indices]
        # Update act_lens
        act_lens = [new_lens[i] for i in indices]
        # Update other variables
        i += 1
        start += L

    # TODO
    # if i != round_len: raise AssertionError(
    #   '!! Counter = {} while round_len = {} (num_steps = {})'.format(
    #     i, round_len, num_steps))

    # Clear dynamic_round_len if necessary
    if is_training: self._clear_dynamic_round_len()

  # endregion : Basic APIs

  # region : Public Methods

  def split(self, *sizes, names=None, over_classes=False, random=False):
    """If over_classes is True, sizes are for each group, and this works only
       for uniform dataset.
    """
    # Sanity check
    if len(sizes) == 0: raise ValueError('!! split sizes not specified')
    elif len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
      # in case this method is used like split([-1, 100, 100])
      #   instead of split(-1, 100, 100)
      sizes = sizes[0]
    if names is not None:
      if not isinstance(names, (tuple, list)):
        raise TypeError('!! names must be a tuple or list of strings')
      if len(names) != len(sizes):
        raise ValueError('!! length of name list and sizes list does not match')
    # Check sizes
    sizes, auto_index, size_accumulator = list(sizes), -1, 0
    for i, size in enumerate(sizes):
      if size is None or size < 0:
        if auto_index < 0:
          auto_index = i
          continue
        else: raise ValueError(
          '!! only one split size can be calculated automatically')
      if not isinstance(size, int) and size < 0:
        raise ValueError('!! size must be a non-negative integer')
      size_accumulator += size

    # Get total size according to over_classes flag
    total_size = self.size
    if over_classes:
      sample_nums = [len(g) for g in self.groups]
      for n in sample_nums: assert n == sample_nums[0]
      total_size = sample_nums[0]
    # Calculate size automatically if necessary
    if auto_index >= 0:
      sizes[auto_index] = total_size - size_accumulator
      if sizes[auto_index] < 0: raise ValueError(
        '!! negative value appears when calculating size automatically')
    elif size_accumulator != total_size: raise ValueError(
      '!! total size does not match size of the data set to split')

    # Split data set
    data_sets, cursor = (), 0
    indices_pool = set(range(self.size))
    if over_classes:
      group_pool = [set(range(total_size)) for _ in self.groups]
    for i, size in enumerate(sizes):
      if size == 0: continue
      # Generate indices
      if not over_classes:
        if not random:
          indices = slice(cursor, cursor + size)
        else:
          indices = np.random.choice(list(indices_pool), size, replace=False)
          indices_pool -= set(indices)
      else:
        indices = []
        if not random:
          for g in self.groups:
            indices += g[slice(cursor, cursor + size)]
        else:
          for j, g in enumerate(group_pool):
            idcs = np.random.choice(list(g), size, replace=False)
            group_pool[j] = g - set(idcs)
            for idx in idcs: indices.append(self.groups[j][idx])
      # Get subset
      data_set = self[indices]
      if names is not None: data_set.name = names[i]
      data_sets += (data_set,)
      cursor += size

    return data_sets


  def refresh_groups(self, target_key='targets'):
    # TODO: this method is overlapping with self.dense_labels property
    #       try to fix it.
    # Sanity check
    if self.num_classes is None:
      raise AssertionError('!! DataSet should have known # classes')
    targets = self[target_key]
    if targets is None:
      raise AssertionError('!! Can not find targets with key `{}`'.format(
        target_key))
    # Handle sequence summary situation
    if isinstance(targets, (list, tuple)):
      targets = np.concatenate(targets, axis=0)
    dense_labels = misc.convert_to_dense_labels(targets)
    groups = []
    for i in range(self.num_classes):
      # Find samples of class i and append to groups
      samples = list(np.argwhere([j == i for j in dense_labels]).ravel())
      groups.append(samples)
    self.properties[self.GROUPS] = groups

  # endregion : Public Methods

  # region : Private Methods

  def _finalize(self, data_set, indices=None):
    assert isinstance(data_set, DataSet)
    data_set.__class__ = self.__class__
    data_set.properties = self.properties.copy()

    if indices is not None:
      for k, v in self.properties.items():
        if isinstance(v, tuple) and len(v) == self.size:
          data_set.properties[k] = self._get_subset(v, indices)

    # Groups should not be passed to subset
    data_set.properties.pop(self.GROUPS, None)
    if self.num_classes is not None and 'targets' in self.data_dict.keys():
      data_set.refresh_groups()
    return data_set

  def _select(self, batch_index, batch_size, upper_bound=None, training=False):
    """The result indices may have a length less than batch_size specified.
       * shuffle option is handled in  gen_[rnn_]batches method
    """
    if upper_bound is None: upper_bound = self.size
    assert isinstance(batch_index, int) and batch_index >= 0
    checker.check_positive_integer(batch_size)

    from_index = batch_index * batch_size
    to_index = min((batch_index + 1) * batch_size, upper_bound)
    indices = list(range(from_index, to_index))

    # return indices
    if training: return self.indices[indices]
    else: return self._ordered_indices[indices]


  def _check_data(self):
    """data_dict should be a non-empty dictionary containing regular numpy
       arrays with the same length"""
    # Make sure data_dict is a non-empty dictionary
    if not isinstance(self.data_dict, dict) or len(self.data_dict) == 0:
      raise TypeError('!! data_dict must be a non-empty dictionary')

    data_length = len(list(self.data_dict.values())[0])

    # Check each item in data_dict
    for name, array in self.data_dict.items():
      # Check type and length
      if not isinstance(array, np.ndarray) or len(array) != data_length:
        raise ValueError('!! {} should be a numpy array with length {}'.format(
          name, data_length))
      # Check sample shape
      if len(array.shape) == 1:
        self.data_dict[name] = np.reshape(array, (-1, 1))

  def _apply(self, f, data_dict=None):
    """Apply callable method f to all data in data_dict. If data_dict is not
       provided, self.data_dict will be used as default"""
    assert callable(f)
    if data_dict is None: data_dict = self.data_dict
    result_dict = {}
    for k, v in data_dict.items(): result_dict[k] = f(v)
    return result_dict

  def _random_from_whole_seq(self, batch_size):
    L = checker.check_positive_integer(hub.random_sample_length)
    assert L < self.size
    # Generate indices for each sequence
    indices = np.random.randint(low=0, high=self.size - L + 1, size=batch_size)
    def f(array):
      data = np.zeros(shape=(batch_size, L, *array.shape[1:]))
      for i, start_j in enumerate(indices):
        data[i] = array[start_j:start_j+L]
      return data
    return DataSet(data_dict=self._apply(f), is_rnn_input=True,
                   name=self.name, **self.properties)

  def _convert_to_rnn_input(self, training, batch_size=1):
    """Used in partitioning a sequence, e.g. partitioning PTB data set.
       Given: Total_length=M, Batch_size=N, overlap_percent=p
       Calculate: Total_num_steps denoted as L
       [(N-1)*(1-p)+1]*L <= M

    """
    assert isinstance(training, bool)
    checker.check_positive_integer(batch_size)
    if training and hub.random_sample_length is not None:
      # TODO: beta branch
      return self._random_from_whole_seq(batch_size)
    # Init a shared indices list
    indices = []
    def f(array):
      assert isinstance(array, np.ndarray) and len(array.shape) > 1
      # Get overlap percent
      M, N, p = len(array), batch_size, hub.overlap_pct if training else 0.
      assert 0 <= p < 1
      L = int(M/((N - 1)*(1 - p) + 1))
      L_bar = int(L*(1 - p))
      data = np.zeros(shape=(batch_size, L, *array.shape[1:]))
      r = hub.random_shift_pct if training else 0.
      assert 0 <= r < 1
      s = int(np.floor(r*L))
      for i in range(batch_size):
        # Use already generated indices if provided
        if len(indices) == batch_size:
          j_start, j_end = indices[i]
        else:
          j_start = i * L_bar
          if s > 0: j_start += np.random.randint(-s, s)
          j_end = j_start + L
          if j_start < 0: j_start, j_end = 0, L
          elif j_end > M: j_start, j_end = M - L, M
          indices.append((j_start, j_end))
        data[i] = array[j_start:j_end]
      return data
    return DataSet(data_dict=self._apply(f), is_rnn_input=True,
                   name=self.name, **self.properties)

  @staticmethod
  def _get_subset(data, indices):
    """Get subset of data. For DataSet, data is np.ndarray.
       For SequenceSet, data is a list.
    """
    if np.isscalar(indices):
      if isinstance(data, (list, tuple)): return [data[indices]]
      elif isinstance(data, np.ndarray):
        subset = data[indices]
        return np.reshape(subset, (1, *subset.shape))
    elif isinstance(indices, (list, tuple, np.ndarray)):
      if isinstance(data, (list, tuple)): return [data[i] for i in indices]
      elif isinstance(data, np.ndarray): return data[np.array(indices)]
    elif isinstance(indices, slice): return data[indices]
    else: raise TypeError('Unknown indices format: {}'.format(type(indices)))

    raise TypeError('Unknown data format: {}'.format(type(data)))

  def _rand_indices(self, upper_bound=None, size=1):
    if upper_bound is None: upper_bound = self.size
    assert self.features is not None
    if not hub.rand_over_classes:
      indices = np.random.randint(upper_bound, size=size)
    else:
      classes = np.random.randint(self.num_classes, size=size)
      indices = []
      for cls in classes:
        group_index = np.random.randint(len(self.groups[cls]))
        indices.append(self.groups[cls][group_index])

    if len(indices) == 1: return int(indices[0])
    else: return indices

  def _init_indices(self, shuffle):
    indices = list(range(len(self.features)))
    if shuffle: np.random.shuffle(indices)
    self.indices = np.array(indices)

  def _set_dynamic_round_len(self, val):
    # To be compatible with old version
    assert getattr(self, '_dynamic_round_len', None) is None
    self._dynamic_round_len = checker.check_positive_integer(val)

  def _clear_dynamic_round_len(self):
    self._dynamic_round_len = None


  @staticmethod
  def _get_dynamic_round_len(act_lens, num_steps, training):
    """
                                 not train
    x x x x x|x x x x x|x x x/x x:x x x x x x x
    x x x x x|x x x x x|x x x/   :
    x x x x x|x x x x x|x x x/x  :
    x x x x x|x x x x x|x x x/x x:x x x x x
                           train
    """
    assert isinstance(act_lens, (np.ndarray, list)) and len(act_lens) > 0
    checker.check_positive_integer(num_steps)
    counter = 0
    while len(act_lens) > 0:
      # Find the shortest sequence
      sl = min(act_lens)
      assert sl > 0
      # Calculate iterations (IMPORTANT). Note that during training act_len
      # .. does not help to avoid inappropriate gradient flow thus sequences
      # .. have to be truncated
      n = int(np.ceil(sl / num_steps))
      counter += n
      # Update act_lens list
      L = sl if training else n * num_steps
      act_lens = [al for al in [al - L for al in act_lens] if al > 0]
    return counter

  # endregion : Private Methods


if __name__ == '__main__':
  features = np.arange(12)
  data_set = DataSet(features)



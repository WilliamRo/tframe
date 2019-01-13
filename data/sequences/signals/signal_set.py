from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import checker
from tframe import pedia
from tframe.data.dataset import DataSet
from tframe.data.sequences.seq_set import SequenceSet

from tframe.data.sequences.signals.tf_signal import Signal

import tframe.utils.misc as misc


class SignalSet(SequenceSet):
  """Container for signals. Signal data should only be stored in
     signals and responses. Otherwise errors may occur while loading
     from local"""

  CONVERTER = 'CONVERTER'

  def __init__(self, signals, responses=None, data_dict=None,
               summ_dict=None, n_to_one=False, name='signal_set1',
               converter=None, **kwargs):
    # Check signals
    signal_dict, fs = self._check_signals(signals, responses)
    data_dict = {} if data_dict is None else data_dict
    data_dict.update(signal_dict)
    kwargs.update({pedia.sampling_frequency: fs})

    # Call parent's constructor
    SequenceSet.__init__(self, data_dict=data_dict, summ_dict=summ_dict,
                         n_to_one=n_to_one, name=name, **kwargs)
    # Attributes
    if converter is not None:
      assert callable(converter)
      self.converter = converter

  # region : Properties

  @property
  def converter(self):
    return self.properties[self.CONVERTER]

  @converter.setter
  def converter(self, val):
    assert callable(val)
    self.properties[self.CONVERTER] = val

  @property
  def structure(self):
    if self.features is not None: return super().structure
    assert isinstance(self.signals, list)
    result = []
    for s in self.signals:
      assert isinstance(s, Signal)
      result.append(len(s))
    return result

  @property
  def signals(self):
    signals = self.data_dict[pedia.signals]
    checker.check_type(signals, Signal)
    return signals

  @property
  def responses(self):
    responses = self.data_dict.get(pedia.responses, None)
    if responses is not None:
      assert len(responses) == len(self.signals)
      checker.check_type(responses, Signal)
    return responses

  @property
  def fs(self):
    fs = self.properties.get(pedia.sampling_frequency, None)
    assert isinstance(fs, (int, float)) and fs > 0
    return fs

  @property
  def as_sequence_set(self):
    return self._convert_to(SequenceSet)

  @property
  def as_data_set(self):
    return self._convert_to(DataSet)

  # endregion : Properties

  # region : Override

  @classmethod
  def load(cls, filename):
    data_set = super().load(filename)
    if not isinstance(data_set, SignalSet):
      raise TypeError('!! Can not resolve data set of type {}'.format(
        type(data_set)))
    # TODO
    # Set fs to each signal
    fs = data_set.fs
    for s in data_set.signals: s.fs = fs
    # Set fs to each response
    responses = data_set.responses
    if responses is not None:
      for r in responses: r.fs = fs
    # Return data set
    return data_set

  # endregion : Override

  # region : Public Methods

  def init_features_and_targets(self, targets_key=None, memory_depth=None,
                                skip_head=True):
    """Initialize features and targets using data in data_dict.
        After initialization, data_dict will be cleared"""
    # If target key is not provided, try to find one in data_dict
    if targets_key is None:
      targets_candidates = None
      for key, val in self.data_dict.items():
        if key != pedia.signals:
          targets_candidates = val
          break
    else: targets_candidates = self.data_dict.get(targets_key)
    # If memory depth is None, init features as signals
    if memory_depth is None:
      self.features = self.signals
      self.targets = targets_candidates
      self._check_data()
      return
    # Initialize features (as causal matrix) and targets one by one
    features = []
    targets = []
    checker.check_positive_integer(memory_depth)
    start_at = memory_depth - 1 if skip_head else 0
    for i, signal in enumerate(self.signals):
      # Append signal to features
      assert isinstance(signal, Signal)
      features.append(signal.causal_matrix(memory_depth, skip_head))
      # Append target
      if targets_candidates is not None:
        target = targets_candidates[i]
        # For response target
        if targets_key == pedia.responses:
          assert isinstance(target, Signal)
          target = target.causal_matrix(memory_depth=1)
          target = target[start_at:]
        elif targets_key == pedia.labels:
          assert isinstance(target, np.ndarray)
          label_len = len(target)
          if label_len == len(signal):
            target = target[start_at:]
          else: assert label_len == 1
        # Append target to targets list
        targets.append(target)
    # Set features and targets
    self.features = features
    self.targets = None if targets_candidates is None else targets
    # Abandon data_dict
    if memory_depth > 1: self.data_dict = {}
    # Check data
    self._check_data()

  def plot(self, index=0, db=True):
    from tframe.data.sequences.signals.figure import Figure, Subplot
    if isinstance(self.signals, Signal): index = range(len(self.signals))
    x = self.signals[index]
    y = None if self.responses is None else self.responses[index]
    fig = Figure('{} Input & Output'.format(self.name))
    fig.add(Subplot.AmplitudeSpectrum(
      x, prefix='{} - Input Signal'.format(self.name), db=db))
    if y is not None:
      fig.add(Subplot.AmplitudeSpectrum(y, prefix='Output Signal', db=db))
    fig.plot()

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _check_signals(signals, responses):
    # Check signals
    if isinstance(signals, Signal): signals = [signals]
    checker.check_type(signals, Signal)
    signal_dict = {}
    # Make sure all signals are sampled under a same fs
    fs = signals[0].fs
    for i in range(1, len(signals)):
      if fs != signals[i].fs: raise ValueError(
        '!! All signals in SignalSet must have the same sampling frequency')
    signal_dict[pedia.signals] = signals
    # Check responses
    if responses is not None:
      if isinstance(responses, Signal): responses = [responses]
      if len(signals) != len(responses): raise ValueError(
        '!! length of responses({}) does not match that of signals({})'.format(
          len(responses), len(signals)))
      checker.check_type(responses, Signal)
      for r in responses:
        if r.fs != fs: raise ValueError(
          '!! All responses must have the same sampling frequency with signals')
      signal_dict[pedia.responses] = responses
    # Return signal dict and fs
    return signal_dict, fs

  def _convert_to(self, _class_=SequenceSet):
    assert _class_ in (SequenceSet, DataSet)
    assert callable(self.converter)
    data_set = self.converter(self)
    assert isinstance(data_set, DataSet)
    data_set.properties[pedia.signals] = data_set.data_dict.pop(pedia.signals)
    if pedia.responses in self.data_dict.keys():
      data_set.properties[pedia.responses] = data_set.data_dict.pop(
        pedia.responses)
    return data_set

  # endregion : Private Methods

  # region : Public Static Methods

  @staticmethod
  def chop_with_stride(x, y, size, stride, rand_shift=True):
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    checker.check_type([size, stride], int)

    out_len = SignalSet.chop_with_stride_len_f(len(x), size, stride)
    x_out = np.zeros(shape=(out_len, size))
    y_out = np.zeros(shape=(out_len, *y.shape[1:]))

    if rand_shift:
      remain = len(x) - ((out_len - 1) * stride + size)
      shift = np.random.randint(remain + 1)
    else: shift = 0
    for i in range(out_len):
      # Fill in x
      x_out[i] = x[shift + stride * i:shift + stride * i + size]
      # Fill in y if necessary
      if len(x) == len(y):
        y_out[i] = y[shift + stride * i:shift + stride * i + size]

    if len(x) != len(y):
      assert len(y) == 1
      y = np.tile(y, (out_len, 1))

    return x_out, y

  @staticmethod
  def chop_with_stride_len_f(length, size, stride):
    checker.check_type([length, size, stride], int)
    assert size >= stride
    return (length - size) // stride + 1

  # endregion : Public Static Methods


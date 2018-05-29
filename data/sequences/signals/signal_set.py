from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import checker
from tframe import pedia
from tframe.data.dataset import DataSet
from tframe.data.sequences.signals.signal import Signal


class SignalSet(DataSet):
  """Container for signals. Signal data should only be stored in
     signals and responses. Otherwise errors may occur while loading
     from local"""

  def __init__(self, signals, responses=None, data_dict=None,
               name='signal_set1', **kwargs):
    # Check signals
    signal_dict, fs = self._check_signals(signals, responses)
    data_dict = {} if data_dict is None else {}
    data_dict.update(signal_dict)
    kwargs.update({pedia.sampling_frequency: fs})
    # Call parent't constructor
    DataSet.__init__(self, data_dict=data_dict, name=name, **kwargs)

  # region : Properties

  @property
  def signals(self):
    signals = self.data_dict[pedia.signals]
    assert isinstance(signals, list)
    checker.check_type(signals, Signal)
    return signals

  @property
  def responses(self):
    responses = self.data_dict.get(pedia.responses, None)
    if responses is not None:
      assert isinstance(responses, list)
      assert len(responses) == len(self.signals)
      checker.check_type(responses, Signal)
    return responses

  @property
  def fs(self):
    fs = self.properties.get(pedia.sampling_frequency, None)
    assert isinstance(fs, (int, float)) and fs > 0
    return fs

  # endregion : Properties

  # region : Override

  @staticmethod
  def load(filename):
    data_set = DataSet.load(filename)
    if not isinstance(data_set, SignalSet):
      raise TypeError('!! Can not resolve data set of type {}'.format(
        type(data_set)))
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

  def init_features_and_targets(self, targets_key=None, memory_depth=1,
                                skip_head=True):
    # Check target key
    if targets_key is None: targets_key = pedia.responses
    targets_candidates = self.data_dict.get(targets_key, None)
    # Initialize features and targets one by one
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
        if targets_key == pedia.responses: target = target[start_at:]
        targets.append(target)
    # Set features and targets
    self.features = features
    self.targets = None if targets_candidates is None else targets
    # Check data
    self._check_data()

  def plot(self, index=0, db=True):
    from tframe.data.sequences.signals.figure import Figure, Subplot
    x = self.signals[index]
    y = None if self.responses is None else self.responses[index]
    fig = Figure('{} Input & Output'.format(self.name))
    fig.add(Subplot.AmplitudeSpectrum(x, prefix='Input Signal', db=db))
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

  # endregion : Private Methods


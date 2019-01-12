from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tframe import console
from tframe import pedia
from tframe import checker
from tframe.data.base_classes import DataAgent
from tframe.data.sequences.signals.tf_signal import Signal
from tframe.data.sequences.signals.signal_set import SignalSet


class WHBM(DataAgent):
  """Wiener-Hammerstein BenchMark"""

  DATA_NAME = 'Wiener-Hammerstein BenchMark'
  DATA_URL = 'http://www.ee.kth.se/~hjalmars/ifac_tc11_benchmarks/2009_wienerhammerstein/WienerHammerBenchMark.mat'
  TFD_FILE_NAME = 'whbm.tfds'

  PROPERTIES = {pedia.sampling_frequency: 0}

  @classmethod
  def load(cls, data_dir, train_size=-1, validate_size=5000, test_size=88000,
           memory_depth=1, skip_head=True):
    data_set = cls.load_as_tframe_data(data_dir)
    assert isinstance(data_set, SignalSet)
    # Initialize data before splitting
    data_set.init_features_and_targets(
      targets_key=pedia.responses, memory_depth=memory_depth,
      skip_head=skip_head)
    # Split data set
    data_sets = data_set[0].split(
      train_size, validate_size, test_size,
      names=('training set', 'validation set', 'test set'))
    # Show data info
    cls._show_data_sets_info(data_sets)
    return data_sets

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    file_path = os.path.join(data_dir, cls.TFD_FILE_NAME)
    if os.path.exists(file_path): return SignalSet.load(file_path)
    # If .tfd file does not exist, try to convert from raw data
    console.show_status('Trying to convert raw data to tframe DataSet ...')
    signal, response = cls.load_as_numpy_arrays(data_dir)
    data_set = SignalSet(signal, response, name=cls.DATA_NAME)
    console.show_status('Successfully converted samples of length {}'.format(
      data_set.size))
    # Save DataSet
    console.show_status('Saving signal data set ...')
    data_set.save(file_path)
    console.show_status('Data set saved to {}'.format(file_path))
    return data_set

  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    """Load a sequence (with double precision) of length 188000 sampled
       under a frequency of 51200 Hz"""
    import scipy.io as sio
    uBenchMark = 'uBenchMark'
    yBenchMark = 'yBenchMark'
    # Check .mat file
    file_path = cls._check_raw_data(data_dir)
    # Load .mat file
    data_dict = sio.loadmat(file_path)
    u = data_dict[uBenchMark]
    y = data_dict[yBenchMark]
    fs = data_dict['fs']
    checker.check_type((u, y, fs), np.ndarray)
    # Wrap signals into Signals
    signal = Signal(u.squeeze(), float(fs))
    response = Signal(y.squeeze(), float(fs))
    # Return u, y, fs
    return signal, response

  @staticmethod
  def evaluate(f, data_set, plot=False):
    if not callable(f): raise AssertionError('!! Input f must be callable')
    checker.check_type(data_set, SignalSet)
    assert isinstance(data_set, SignalSet)
    if data_set.targets is None:
      raise ValueError('!! Responses not found in SignalSet')
    u, y = data_set.features, np.ravel(data_set.targets)
    assert isinstance(y, Signal)
    # Show status
    console.show_status('Evaluating {} ...'.format(data_set.name))
    # In evaluation, the sum of each metric is started at t = 1000 instead of
    #  t = 0 to eliminate the influence of transient errors at the beginning of
    #  the simulation
    start_at = 1000
    model_output = Signal(f(u), fs=y.fs)
    delta = y - model_output
    err = delta[start_at:]
    assert isinstance(err, Signal)
    ratio = lambda val: 100.0 * val / y.rms

    # The mean value of the simulation error in time domain
    val = err.average
    console.supplement('E[err] = {:.4f}mV ({:.3f}%)'.format(
      val * 1000, ratio(val)))
    # The standard deviation of the error in time domain
    val = float(np.std(err))
    console.supplement('STD[err] = {:.4f}mV ({:.3f}%)'.format(
      val * 1000, ratio(val)))
    # The root mean square value of the error in time domain
    val = err.rms
    console.supplement('RMS[err] = {:.4f}mV ({:.3f}%)'.format(
      val * 1000, ratio(val)))

    # Plot
    if not plot: return
    from tframe.data.sequences.signals.figure import Figure, Subplot
    fig = Figure('Simulation Error')
    # Add ground truth
    prefix = 'System Output, $||y|| = {:.4f}$'.format(y.norm)
    fig.add(Subplot.PowerSpectrum(y, prefix=prefix))
    # Add model output
    prefix = 'Model Output, RMS($\Delta$) = ${:.4f}mV$'.format(1000 * err.rms)
    fig.add(Subplot.PowerSpectrum(model_output, prefix=prefix, Error=delta))
    # Plot
    fig.plot()


if __name__ == '__main__':
  data_dir = '../../../examples/whbm/data'
  train_set, test_set = WHBM.load(data_dir, validate_size=0)
  assert isinstance(train_set, SignalSet)
  assert isinstance(test_set, SignalSet)
  train_set.plot()
  test_set.plot()

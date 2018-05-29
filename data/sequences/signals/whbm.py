from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tframe import console
from tframe import pedia
from tframe import checker
from tframe.data.base_classes import DataAgent
from tframe.data.sequences.signals.signal import Signal
from tframe.data.sequences.signals.signal_set import SignalSet


class WHBM(DataAgent):
  """Wiener-Hammerstein BenchMark"""

  DATA_NAME = 'Wiener-Hammerstein BenchMark'
  DATA_URL = 'http://www.ee.kth.se/~hjalmars/ifac_tc11_benchmarks/2009_wienerhammerstein/WienerHammerBenchMark.mat'
  TFD_FILE_NAME = 'whbm.tfds'

  PROPERTIES = {pedia.sampling_frequency: 0}

  @classmethod
  def load(cls, data_dir, train_size=-1, validate_size=5000, test_size=88000):
    data_set = cls.load_as_tframe_data(data_dir)

    data_sets = (None, None)
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


if __name__ == '__main__':
  data_dir = '../../../examples/whbm/data'
  data_set = WHBM.load_as_tframe_data(data_dir)
  data_set.plot()

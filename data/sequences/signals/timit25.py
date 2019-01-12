from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from collections import OrderedDict

from tframe import checker
from tframe import console
from tframe import pedia
from tframe.utils import misc
from tframe.data.base_classes import DataAgent
from tframe.data.sequences.signals.tf_signal import Signal
from tframe.data.sequences.signals.signal_set import SignalSet
from tframe.data.sequences.seq_set import SequenceSet


def clusters2list(clusters):
  result = []
  for cluster in clusters: result.extend(list(cluster))
  return result

class TIMIT25(DataAgent):
  """Totally 25 classes of audio signals each of which has 7
     different examplars. These 25 classes are arranged in 5 clusters based on
     their suffix.

     References:
     [1] Jan Koutnik, etc. A Clockwork RNN. 2014.
     [2] Garofolo, etc. DARPA TIMIT acoustic phonetic continuous corpus CD-ROM,
         1993.
  """
  DATA_NAME = 'TIMIT-25'
  TFD_FILE_NAME = 'timit-25.tfds'

  CLUSTERS = (
    ('making', 'walking', 'cooking', 'looking', 'working'),
    ('biblical', 'cyclical', 'technical', 'classical', 'critical'),
    ('tradition', 'addition', 'audition', 'recognition', 'competition'),
    ('musicians', 'discussions', 'regulations', 'accusations', 'conditions'),
    ('subway', 'leeway', 'freeway', 'highway', 'hallway'),
  )

  PROPERTIES = {
    pedia.classes: clusters2list(CLUSTERS),
    SignalSet.NUM_CLASSES: 25,
  }

  @classmethod
  def load(cls, data_dir, train_size_foreach, raw_data_dir='TIMIT25', **kwargs):

    train_set, test_set = None, None
    return train_set, test_set

  @classmethod
  def load_as_tframe_data(cls, data_dir, file_name=None, raw_data_dir=None):
    # Check file_name
    if file_name is None: file_name = 'timit-25.tfds'
    data_path = os.path.join(data_dir, file_name)
    if os.path.exists(data_path): return SignalSet.load(data_path)
    # If data does not exist, create a new data set
    console.show_status('Loading data ...')
    if raw_data_dir is None:
      raw_data_dir = os.path.join(data_dir, 'TIMIT25')
    data_dict, sr = cls.load_as_numpy_arrays(raw_data_dir)
    console.show_status('Wrapping data into signal set ...')
    signals = []
    targets = []
    groups = []
    signal_index = 0
    for i, word in enumerate(cls.PROPERTIES[pedia.classes]):
      group_indices = []
      target = misc.convert_to_one_hot(
        [i], cls.PROPERTIES[SignalSet.NUM_CLASSES])
      for array in data_dict[word]:
        signals.append(Signal(array, sr))
        targets.append(target)
        group_indices.append(signal_index)
        signal_index += 1
      groups.append(group_indices)
    data_set = SignalSet(
      signals, summ_dict={'targets': targets}, n_to_one=True, name='TIMIT25',
      converter=None, **cls.PROPERTIES)
    data_set.properties[data_set.GROUPS] = groups
    data_set.converter = lambda: cls.converter(data_set)
    data_set.batch_preprocessor = cls.preprocessor
    data_set.save(data_path)
    console.show_status('Data set saved to `{}`'.format(data_path))
    return data_set

  @classmethod
  def load_as_numpy_arrays(cls, raw_data_dir):
    """data_dir should contains 25 folders corresponding to 25 different
       words(classes), each of which should contain 7 different examples.
       File names of examples from the same class should be [1-7].wav

    :return an OrderedDict with structure {word1: [seq1, ..., seq7], ...}
    """
    # Sampling rate is 16000 according to `https://catalog.ldc.upenn.edu/LDC93S1`
    sampling_rate = 16000

    # Try to import librosa
    try: import librosa
    except: raise ImportError(
      '!! Can not import librosa. You should install this ' 
      'package before reading .wav files.')

    # Read data
    timit25 = OrderedDict()
    for cluster in cls.CLUSTERS:
      for word in cluster:
        timit25[word] = []
        console.show_status('Reading `{}`'.format(word))
        path = os.path.join(raw_data_dir, '{}'.format(word))
        for i in range(7):
          wav_path = os.path.join(path, '{}.wav'.format(i + 1))
          data, _ = librosa.core.load(wav_path, sr=sampling_rate)
          timit25[word].append(data)

    return timit25, sampling_rate

  @classmethod
  def converter(cls, signal_set):
    """Convert signal set to sequence set. See Jan Koutnic, etc (2014):
       `Each sequence element consists of 12-dimensional MFCC vector with
        a pre-emphasis coefﬁcient of 0.97. Each of the 13 channels was then
        normalized to have zero mean and unit variance over the whole
        training set.`
    """
    assert isinstance(signal_set, SignalSet)
    
    signal_set.features = None
    return signal_set

  @classmethod
  def preprocessor(cls, data_set, is_training):
    if not is_training: return data_set
    assert isinstance(data_set, SequenceSet)
    return data_set


if __name__ == '__main__':
  data_dir = r'E:\rnn_club\03-TIMIT\data'
  raw_data_dir = r'E:\rnn_club\03-TIMIT\data\TIMIT25'
  data_set = TIMIT25.load_as_tframe_data(data_dir)
  s1, s2 = data_set.split(5, 2, over_classes=True, random=True)
  _ = None


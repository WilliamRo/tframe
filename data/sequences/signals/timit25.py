from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from collections import OrderedDict

import tframe.utils.maths.dsp as dsp

from tframe import checker
from tframe import console
from tframe import context
from tframe import local
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

  # Sampling rate is 16000 according to `https://catalog.ldc.upenn.edu/LDC93S1`
  SAMPLING_RATE = 16000

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
  def load(cls, data_dir, num_train_foreach, raw_data_dir='TIMIT25',
           random=True, **kwargs):
    signal_set = cls.load_as_tframe_data(data_dir)
    signal_set = signal_set.as_sequence_set
    return signal_set.split(
      num_train_foreach, None, names=('train_set', 'test_set'),
      over_classes=True, random=random)

  @classmethod
  def load_as_tframe_data(cls, data_dir, file_name=None, raw_data_dir=None,
                          force_create=False):
    # Check file_name
    if file_name is None: file_name = 'timit-25.tfds'
    data_path = os.path.join(data_dir, file_name)
    if not force_create and os.path.exists(data_path):
      return SignalSet.load(data_path)
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
      converter=cls.converter, **cls.PROPERTIES)
    data_set.properties[data_set.GROUPS] = groups
    data_set.batch_preprocessor = cls.preprocessor
    if not force_create:
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
    # Read data
    timit25 = OrderedDict()
    for cluster in cls.CLUSTERS:
      for word in cluster:
        timit25[word] = []
        console.show_status('Reading `{}`'.format(word))
        path = os.path.join(raw_data_dir, '{}'.format(word))
        for i in range(7):
          wav_path = os.path.join(path, '{}.wav'.format(i + 1))
          data, sr = local.load_wav_file(wav_path)
          assert sr == cls.SAMPLING_RATE
          timit25[word].append(data)

    return timit25, cls.SAMPLING_RATE

  @classmethod
  def converter(cls, signal_set):
    """Convert signal set to sequence set. See Jan Koutnic, etc (2014):
       `Each sequence element consists of 12-dimensional MFCC vector with
        a pre-emphasis coefficient of 0.97. Each of the 13 channels was then
        normalized to have zero mean and unit variance over the whole
        training set.`
    """
    assert isinstance(signal_set, SignalSet)
    # Try to import librosa
    try: import librosa
    except: raise ImportError(
      '!! Can not import librosa. You should install this '
      'package before reading .wav files.')

    # Specify arguments
    pre_emp_coef = 0.97
    n_fft = int(cls.SAMPLING_RATE / 1000 * 25)
    hop_length = int(cls.SAMPLING_RATE / 1000 * 10)
    features = []
    for signal_ in signal_set.signals:
      assert isinstance(signal_, Signal)
      # Do pre-emphasis
      signal_ = dsp.pre_emphasize(signal_, pre_emp_coef)
      # Generate 12 channels using MFCC
      mfcc12 = librosa.feature.mfcc(
        signal_, sr=cls.SAMPLING_RATE, n_mfcc=12,
        n_fft=n_fft, hop_length=hop_length)
      # Transpose mfcc matrix to shape (length, 12)
      mfcc12 = np.transpose(mfcc12)

      # Calculate energy
      energy = dsp.short_time_energy(signal_, n_fft, stride=hop_length)
      energy = energy.reshape((-1, 1))
      assert mfcc12.shape[0] == energy.shape[0]
      feature = np.concatenate((mfcc12, energy), axis=1)
      # Append to feature
      features.append(feature)

    # Calculate mean and variance for each channel
    stack = np.concatenate(features)
    mean = np.mean(stack, axis=0)
    sigma = np.std(stack, axis=0)
    # Normalize each channel
    for i, array in enumerate(features):
      features[i] = (array - mean) / sigma

    signal_set.features = features
    return signal_set

  @classmethod
  def preprocessor(cls, data_set, is_training):
    if not is_training: return data_set
    assert isinstance(data_set, SequenceSet)
    sigma = 0.6
    for i, input_ in enumerate(data_set.features):
      data_set.features[i] = input_ + np.random.randn(*input_.shape) * sigma
    return data_set

  @classmethod
  def evaluate(cls, trainer, data_set):
    from tframe.trainers.trainer import Trainer
    assert isinstance(trainer, Trainer)
    assert isinstance(data_set, SequenceSet)

    # Load best model
    if trainer.th.save_model:
      flag, _, _ = trainer.model.agent.load()
      assert flag
    else: console.warning('Save model option should be turned on.')

    console.show_status('Evaluating on test set ...')
    metric_dict = trainer.model.validate_model(data_set)
    accuracy = 100 * metric_dict[trainer.model.metrics_manager.eval_slot]
    err = 100 - accuracy
    msg = 'Error % on test set is {:.2f}%'.format(err)
    console.show_status(msg)
    trainer.model.agent.put_down_criterion('Error %', err)
    trainer.model.agent.take_notes(msg)


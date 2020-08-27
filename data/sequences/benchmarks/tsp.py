from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from collections import OrderedDict

from tframe import checker
from tframe import console
from tframe.data.base_classes import DataAgent
from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.perpetual_machine import PerpetualMachine


def engine(number, N=3, T_or_interval=100, var_x=0.2, add_noise=False,
           var_y=0.1):
  # Check input
  if isinstance(T_or_interval, int):
    L_min = T_or_interval
    L_max = int(np.round(L_min * 1.1))
  else:
    checker.check_type(T_or_interval, int)
    L_min, L_max = T_or_interval
  checker.check_positive_integer(N)
  assert number in (1, -1) and 0 < N <= L_min <= L_max
  # Decide the length
  L = np.random.randint(L_min, L_max + 1)
  sequence = np.random.randn(L) * np.sqrt(var_x)
  sequence[:N] = number
  target = (number + 1.) / 2.
  if add_noise: target -= number * 0.2 - np.random.randn() * np.sqrt(var_y)
  return sequence, np.array([target])


class TSP(DataAgent):
  """Two Sequence Problem"""
  DATA_NAME = 'TwoSequenceProblem'

  @classmethod
  def engine(cls, N, T, var_x, noisy, var_y):
    def _engine(size):
      return cls._get_one_data_set(size, N, T, var_x, noisy, var_y)
    return _engine

  @classmethod
  def load(cls, data_dir, validate_size=512, test_size=2560, N=3, T=100,
           var_x=0.2, add_noise=False, var_y=0.1, **kwargs):
    # Load train set
    train_set = PerpetualMachine(
      'TSPPM', cls.engine(N, T, var_x, add_noise, var_y))
    # Load validation set and test set
    load_as_tfd = lambda size, prefix, vary=var_y: cls.load_as_tframe_data(
      data_dir, size, N=N, T=T, var_x=var_x, add_noise=add_noise,
      var_y=vary, prefix=prefix)
    val_set = load_as_tfd(validate_size, 'val_', vary=0.)
    test_set = load_as_tfd(test_size, 'test_', vary=0.)
    return train_set, val_set, test_set

  @classmethod
  def load_as_tframe_data(cls, data_dir, size=2560, file_name=None, N=3, T=100,
                          var_x=0.2, add_noise=False, var_y=0.1, prefix=''):
    # Check file_name
    if file_name is None:
      file_name = cls._get_file_name(size, N, T, var_x, add_noise, var_y)
      file_name = prefix + file_name + '.tfds'
    data_path = os.path.join(data_dir, file_name)
    if os.path.exists(data_path): return SequenceSet.load(data_path)
    # If data does not exist, create a new data set
    console.show_status('Creating data ...')
    data_set = cls._get_one_data_set(size, N, T, var_x, add_noise, var_y)
    console.show_status('Saving data set ...')
    data_set.save(data_path)
    console.show_status('Data set saved to `{}`'.format(data_path))
    return data_set

  @classmethod
  def _get_one_data_set(cls, size, N, T, var_x, noisy, var_y):
    features, targets = [], []
    for _ in range(size):
      number = np.random.choice([-1, 1])
      x, y = engine(number, N, T, var_x, noisy, var_y)
      features.append(x)
      targets.append(y)
    # Wrap data into a SequenceSet
    data_set = SequenceSet(
      features, summ_dict={'targets': targets}, n_to_one=True,
      name='Noisy Sequences' if noisy else 'Noise-free Sequences',
      N=N, T=T, var_x=var_x, noisy=noisy, var_y=var_y)
    return data_set

  @classmethod
  def _get_file_name(cls, size, N, T, var_x, add_noise, var_y):
    checker.check_positive_integer(N)
    checker.check_positive_integer(T)
    file_name = '{}_{}_N{}T{}_vx{}_{}'.format(
      'TSP', size, N, T, var_x, 'noisy' if add_noise else 'noise-free')
    if add_noise: file_name += '_vy{}'.format(var_y)
    return file_name

  # region : Probe Methods

  @staticmethod
  def amu18(val_set, test_set, trainer):
    """Probe method accepts trainer as the only parameter"""
    # Do some importing
    from tframe import hub
    from tframe.trainers.trainer import Trainer
    from tframe.models.sl.classifier import Classifier
    from tframe.data.sequences.seq_set import SequenceSet
    # Sanity check
    assert isinstance(trainer, Trainer)
    model = trainer.model
    assert isinstance(model, Classifier)
    assert isinstance(val_set, SequenceSet) and val_set.size == 256
    assert isinstance(test_set, SequenceSet) and test_set.size == 2560
    SATISFY_C1 = 'SATISFY_C1'
    BEST_ACC = 'BEST_ACC'
    BEST_MAE = 'BEST_MAE'
    if SATISFY_C1 not in val_set.properties.keys():
      val_set.properties[SATISFY_C1] = False

    # Calculate accuracy and MAE on val_set
    predictions = model.predict(val_set, batch_size=-1)
    predictions = np.array([q[0] for q in predictions])
    labels = np.array([p[0] for p in val_set.summ_dict['targets']])
    assert hub.prediction_threshold > 0
    abs_deltas = abs(predictions - labels)
    trues =  abs_deltas < hub.prediction_threshold
    accuracy = sum(trues) / len(trues)
    MAE = np.average(abs_deltas)

    # Put down criterion if records appear
    if (BEST_ACC not in val_set.properties.keys()
        or val_set[BEST_ACC] < accuracy):
      val_set.properties[BEST_ACC] = accuracy
      trainer.model.agent.put_down_criterion('Best Acc', accuracy)

    if (BEST_MAE not in val_set.properties.keys()
        or val_set[BEST_MAE] > MAE):
      val_set.properties[BEST_MAE] = MAE
      trainer.model.agent.put_down_criterion('Best MAE', MAE)

    # Export tensors if necessary
    if hub.export_tensors_to_note:
      TSP.export_tensors(trainer, accuracy, MAE)

    # Test C1
    if not val_set.properties[SATISFY_C1] and accuracy == 1.0:
      model.agent.take_notes('C1 is satisfied after {} sequences.'.format(
        trainer.counter))
      val_set.properties[SATISFY_C1] = True
      # Take down C1 counter
      model.agent.put_down_criterion('C1', trainer.counter)

    assert hub.epsilon > 0
    if accuracy == 1.0 and MAE < hub.epsilon:
      # Force terminate
      trainer.th.force_terminate = True
      model.agent.take_notes(
        'Criterion is satisfied after {} sequences.'.format(trainer.counter))
      model.agent.put_down_criterion('C2', trainer.counter)

      # Test model on test set
      result = model.validate_model(test_set, batch_size=-1)
      assert isinstance(result, dict) and len(result) == 1
      test_acc = list(result.values())[0]
      misclassification = 1 - test_acc
      msg = 'Misclassification = {:.6f}'.format(misclassification)
      model.agent.take_notes(msg)
      model.agent.put_down_criterion('Misclassification', misclassification)
    else: msg = 'Accuracy = {:.2f}%, MAE = {:.3f}'.format(accuracy * 100, MAE)
    return msg

  @staticmethod
  def export_tensors(trainer, accuracy, MAE):
    scalars = OrderedDict()
    scalars['Loss'] = trainer.loss_history.running_average
    scalars['Accuracy'] = accuracy
    scalars['MAE'] = MAE
    tensors = OrderedDict()
    agent = trainer.model.agent
    agent.take_down_scalars_and_tensors(scalars, tensors)

  # endregion : Probe Methods


if __name__ == '__main__':
  train_set, val_set, test_set = TSP.load('E:/tmp')


import numpy as np

from tframe import checker
from tframe import Predictor
from tframe.data.dataset import DataSet
from tframe.data.sequences.signals.signal_set import SignalSet
from tframe.data.sequences.signals.whbm import WHBM


def load_data(path, memory_depth=1, validate_size=5000, test_size=88000):
  data_sets = WHBM.load(
    path, validate_size=validate_size, test_size=test_size,
    memory_depth=memory_depth, skip_head=True)
  checker.check_type(data_sets, SignalSet)
  return data_sets


def evaluate(model, data_set, plot=False):
  def f(u):
    assert isinstance(model, Predictor)
    return np.ravel(model.predict(DataSet(features=u)))
  return WHBM.evaluate(f, data_set, plot)


if __name__ == '__main__':
  train_set, test_set = load_data('./data', validate_size=0)
  assert isinstance(train_set, SignalSet)
  assert isinstance(test_set, SignalSet)
  train_set.plot(train_set)

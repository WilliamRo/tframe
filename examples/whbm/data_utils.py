from tframe import DataSet
from tframe.data.images.mnist import MNIST


def load_data(path):
  train_set, val_set, test_set = MNIST.load(
    path, validate_size=5000, test_size=10000, flatten=False, one_hot=True)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)
  return train_set, val_set, test_set

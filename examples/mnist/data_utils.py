from tframe import DataSet


def load_data(path):
  train_set, val_set, test_set = None, None, None
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)
  return train_set, val_set, test_set

from tframe import TFData


def load_data(path):
  train_set, val_set, test_set = None, None, None
  assert isinstance(train_set, TFData)
  assert isinstance(val_set, TFData)
  assert isinstance(test_set, TFData)
  return train_set, val_set, test_set

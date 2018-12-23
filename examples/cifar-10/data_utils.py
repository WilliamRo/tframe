from tframe import DataSet
from tframe.data.images.cifar10 import CIFAR10


def load_data(path):
  train_set, val_set, test_set = CIFAR10.load(
    path, validate_size=10000, test_size=10000, flatten=False, one_hot=True)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)
  return train_set, val_set, test_set


if __name__ == '__main__':
  from tframe.data.images.image_viewer import ImageViewer

  data_path = './data'
  train_set, val_set, test_set = load_data(data_path)

  viewer = ImageViewer(train_set)
  viewer.show()



from tframe import DataSet
from tframe.data.images.mnist import MNIST


def load_data(path):
  train_set, val_set, test_set = MNIST.load(
    path, validate_size=5000, test_size=10000, flatten=False, one_hot=True)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)
  return train_set, val_set, test_set


if __name__ == '__main__':
  from tframe.data.images.image_viewer import ImageViewer
  from mn_core import th
  train_set, val_set, test_set = load_data(th.data_dir)
  viewer = ImageViewer(test_set)
  viewer.show()

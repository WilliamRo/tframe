from tframe import DataSet
from tframe.data.images.mnist import MNIST


def load_data(path, test_directly=False):
  if test_directly:
    train_set, test_set = MNIST.load(
      path, validate_size=0, test_size=10000, flatten=False, one_hot=True)
    assert isinstance(train_set, DataSet)
    assert isinstance(test_set, DataSet)
    return train_set, test_set
  else:
    train_set, val_set, test_set = MNIST.load(
      path, validate_size=5000, test_size=10000, flatten=False, one_hot=True)
    assert isinstance(train_set, DataSet)
    assert isinstance(val_set, DataSet)
    assert isinstance(test_set, DataSet)
    return train_set, val_set, test_set


if __name__ == '__main__':
  from tframe.data.images.image_viewer import ImageViewer
  from mn_core import th
  # train_set, val_set, test_set = load_data('data/')
  train_set, test_set = load_data(th.data_dir, test_directly=True)
  viewer = ImageViewer(test_set)
  viewer.show()

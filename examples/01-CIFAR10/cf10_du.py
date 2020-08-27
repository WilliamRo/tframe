from tframe import DataSet
from tframe.data.images.cifar10 import CIFAR10


def pre(data_batch, is_training):
  assert isinstance(data_batch, DataSet)
  if not is_training: return data_batch

  for i in range(data_batch.size):
    im = data_batch.features[i]
    data_batch.features[i] = im

  return data_batch


def load_data(path):
  train_set, val_set, test_set = CIFAR10.load(
    path, train_size=None, validate_size=5000, test_size=10000,
    flatten=False, one_hot=True)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)
  train_set.batch_preprocessor = pre
  return train_set, val_set, test_set


if __name__ == '__main__':
  from tframe.data.images.image_viewer import ImageViewer
  train_set, val_set, test_set = load_data('data/')
  viewer = ImageViewer(train_set)
  viewer.show()



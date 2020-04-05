import tensorflow as tf

from tframe import Classifier
from tframe.layers import Input, Linear, Activation, Flatten
# from tframe.layers.advanced import Dense
from tframe.layers.normalization import BatchNormalization
from tframe.layers.preprocess import Normalize
from tframe.configs.config_base import Config

from tframe.layers.common import Dropout
from tframe.layers.normalization import BatchNormalization
from tframe.layers.convolutional import Conv2D
from tframe.layers.pooling import MaxPool2D
from tframe.layers.highway import LinearHighway

from tframe.layers.hyper.dense import Dense


def get_container(th, flatten=False):
  assert isinstance(th, Config)
  model = Classifier(mark=th.mark)
  model.add(Input(sample_shape=th.input_shape))
  if th.centralize_data: model.add(Normalize(mu=th.data_mean, sigma=255.))
  if flatten: model.add(Flatten())
  return model


def finalize(th, model):
  assert isinstance(th, Config) and isinstance(model, Classifier)
  # Add output layer
  model.add(Dense(num_neurons=th.num_classes, prune_frac=0.05))
  # model.add(Dense(num_neurons=th.num_classes))
  model.add(Activation('softmax'))
  # Build model
  model.build(metric=['accuracy', 'loss'], batch_metric='accuracy',
              eval_metric='accuracy')
  return model


# TODO: to be deprecated
def multinput(th):
  assert isinstance(th, Config)
  model = Classifier(mark=th.mark)

  # Add hidden layers
  assert isinstance(th.fc_dims, list)
  subnet = model.add(inter_type=model.CONCAT)
  for dims in th.fc_dims:
    subsubnet = subnet.add()
    # Add input layer
    subsubnet.add(Input(sample_shape=th.input_shape))
    subsubnet.add(Flatten())
    assert isinstance(dims, list)

    for dim in dims:
      subsubnet.add(Linear(output_dim=dim))
      # if cf10_core.use_bn: subsubnet.add(BatchNormalization())
      subsubnet.add(Activation(th.actype1))

  # Add output layer
  model.add(Linear(output_dim=th.num_classes))

  # Build model
  model.build(metric=['accuracy', 'loss'], batch_metric='accuracy',
              eval_metric='accuracy')

  return model

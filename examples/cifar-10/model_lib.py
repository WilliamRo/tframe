import tensorflow as tf

from tframe import Classifier
from tframe.layers import Input, Linear, Activation, Flatten
from tframe.layers.normalization import BatchNormalization
from tframe.config import Config

import core


def mlp(th):
  assert isinstance(th, Config)
  # Initiate a model
  model = Classifier(mark=th.mark)

  # Add input layer
  model.add(Input(sample_shape=th.input_shape))
  model.add(Flatten())
  # Add hidden layers
  assert isinstance(th.fc_dims, list)
  for dim in th.fc_dims:
    model.add(Linear(output_dim=dim))
    model.add(BatchNormalization())
    model.add(Activation(th.actype1))

  # Add output layer
  model.add(Linear(output_dim=th.num_classes))
  model.add(Activation('softmax'))

  # Build model
  optimizer=tf.train.AdamOptimizer(learning_rate=th.learning_rate)
  model.build(optimizer=optimizer)

  return model


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
      if core.use_bn: subsubnet.add(BatchNormalization())
      subsubnet.add(Activation(th.actype1))

  # Add output layer
  model.add(Linear(output_dim=th.num_classes))

  # Build model
  optimizer=tf.train.AdamOptimizer(learning_rate=th.learning_rate)
  model.build(optimizer=optimizer)

  return model

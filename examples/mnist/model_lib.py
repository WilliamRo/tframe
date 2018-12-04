import tensorflow as tf

from tframe import Classifier
from tframe.layers import Input, Linear, Activation, Flatten
from tframe.layers.normalization import BatchNormalization
from tframe.configs.config_base import Config


def mlp(th):
  assert isinstance(th, Config)
  # Initiate a model
  model = Classifier(mark=th.mark)

  # Add input layer
  model.add(Input(sample_shape=th.input_shape))
  model.add(Flatten())
  # Add hidden layers
  for _ in range(th.num_blocks):
    model.add(Linear(output_dim=th.hidden_dim))
    model.add(BatchNormalization())
    model.add(Activation(th.actype1))
  # Add output layer
  model.add(Linear(output_dim=th.num_classes))
  model.add(Activation('softmax'))

  # Build model
  optimizer=tf.train.AdamOptimizer(learning_rate=th.learning_rate)
  model.build(optimizer=optimizer)

  return model



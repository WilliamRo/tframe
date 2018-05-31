import tensorflow as tf

from tframe import Predictor
from tframe.layers import Input, Linear, Activation
from tframe.config import Config


def mlp(th):
  assert isinstance(th, Config)
  # Initiate a model
  model = Predictor(mark=th.mark)

  # Add input layer
  model.add(Input(sample_shape=[th.memory_depth]))
  # Add hidden layers
  for _ in range(th.num_blocks):
    model.add(Linear(output_dim=th.hidden_dim))
    model.add(Activation(th.actype1))
  # Add output layer
  model.add(Linear(output_dim=1))

  # Build model
  optimizer=tf.train.AdamOptimizer(learning_rate=th.learning_rate)
  model.build_as_regressor(optimizer)


  return model





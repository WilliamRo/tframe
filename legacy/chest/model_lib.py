from tframe import pedia
from tframe import Predictor, Classifier

from tframe.layers import Input, Linear, Activation
from tframe.nets.rnn_cells import BasicRNNCell
from tframe.models import Recurrent

from tframe.config import Config


def example_model(th):
  assert isinstance(th, Config)
  # Initiate a model
  model = Predictor(mark=th.mark)

  # Add layers
  model.add(Input(sample_shape=[100]))
  # Add hidden layers
  # model.add(...)
  model.add(Linear(output_dim=1))

  # Build model
  model.build()

  return model



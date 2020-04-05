from tframe import monitor
from tframe import Classifier
from tframe.layers import Input, Activation, Flatten
# from tframe.layers.advanced import Dense
from tframe.layers.hyper.dense import Dense
from tframe.configs.config_base import Config
from tframe.layers.preprocess import Normalize

import mn_du


def get_container(th, flatten=False):
  assert isinstance(th, Config)
  model = Classifier(mark=th.mark)
  model.add(Input(sample_shape=th.input_shape))
  model.add(Normalize(sigma=255.))
  if th.centralize_data: model.add(Normalize(mu=th.data_mean))
  if flatten:
    model.add(Flatten())
    # Register extractor and researcher
    model.register_extractor(mn_du.MNIST.connection_heat_map_extractor)
    monitor.register_grad_researcher(mn_du.MNIST.flatten_researcher)
  return model


def finalize(th, model, add_output_layer=True):
  assert isinstance(th, Config) and isinstance(model, Classifier)
  # Add output layer
  if add_output_layer:
    # if th.use_bit_max:
    #   model.add(BitMax(num_classes=th.num_classes, use_softmax=th.use_softmax))
    # else:
    model.add(Dense(num_neurons=th.num_classes, prune_frac=0.1))
    model.add(Activation('softmax'))
  # Build model
  # model.build(th.get_optimizer(), metric=['loss', 'accuracy'],
  model.build(th.get_optimizer(), metric=['accuracy', 'loss'],
              batch_metric='accuracy', eval_metric='accuracy')
  return model


def typical(th, layers, flatten=False):
  assert isinstance(th, Config)
  model = get_container(th, flatten)
  if not isinstance(layers, (list, tuple)): layers = [layers]
  for layer in layers: model.add(layer)
  return finalize(th, model)


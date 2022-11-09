from tframe import console
from tframe import hub as th
from tframe import mu



def get_FNN_predictor(funcs: list, input_shape: list):
  model = mu.Predictor(mark='99-01-01')
  model.add(mu.Input(sample_shape=input_shape))
  for f in funcs: model.add(f)
  model.build(loss='mse')
  return model


def test_hyper_conv1D():
  funcs = [
    mu.HyperConv1D(8, 3, strides=1, padding='same', dilations=1),
    mu.HyperConv1D(16, 3, strides=2, padding='same', dilations=1),
    mu.HyperDeconv1D(8, 3, strides=2, padding='same', dilations=1),
  ]
  return get_FNN_predictor(funcs, input_shape=[64, 1])


def test_hyper_conv2D():
  funcs = [
    mu.HyperConv2D(8, 3, strides=1, padding='same', dilations=1),
    mu.HyperConv2D(16, 3, strides=2, padding='same', dilations=1),
    mu.HyperDeconv2D(8, 3, strides=2, padding='same', dilations=1),
  ]
  return get_FNN_predictor(funcs, input_shape=[64, 64, 1])


def test_hyper_conv3D():
  funcs = [
    mu.HyperConv3D(8, 3, strides=1, padding='same', dilations=1),
    mu.HyperConv3D(16, 3, strides=2, padding='same', dilations=1),
    mu.HyperDeconv3D(8, 3, strides=2, padding='same', dilations=1),
  ]
  return get_FNN_predictor(funcs, input_shape=[64, 64, 64, 1])



if __name__ == '__main__':
  console.suppress_logging()

  th.save_model = False

  X = 3
  model = [test_hyper_conv1D,
           test_hyper_conv2D,
           test_hyper_conv3D][X - 1]()

  model.rehearse(export_graph=False, build_model=False, mark='99-01-01')

from tframe import tf

import tframe as tfr


def with_graph(meth):
  def wrapper(*args, **kwargs):
    obj = args[0]
    assert hasattr(obj, 'graph')
    with obj.graph.as_default():
      return meth(*args, **kwargs)
  return wrapper


def with_graph_if_has(meth):
  def wrapper(*args, **kwargs):
    obj = args[0]
    if hasattr(obj, 'graph'):
      with obj.graph.as_default():
        return meth(*args, **kwargs)
    else:
      return meth(*args, **kwargs)
  return wrapper


def init_with_graph(init):
  def wrapper(*args, **kwargs):
    graph = tfr.context.current_graph
    assert isinstance(graph, tf.Graph)
    with graph.as_default(): init(*args, **kwargs)
  return wrapper


def single_input(_link):
  from tframe.layers.layer import Layer
  from tframe.nets.net import Net

  def wrapper(*args):
    assert isinstance(args[0], (Layer, Net))
    input_ = args[1]
    if isinstance(input_, list):
      if len(input_) != 1:
        raise ValueError('!! This function only accept single input')
      input_ = input_[0]
    if not isinstance(input_, tf.Tensor):
      raise TypeError('!! This layer only accept a Tensor as input')
    args = (args[0], input_) + args[2:]
    return _link(*args)

  return wrapper

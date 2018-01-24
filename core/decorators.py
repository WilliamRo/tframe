import tensorflow as tf
import tframe as tfr


def with_graph(meth):
  def wrapper(*args, **kwargs):
    obj = args[0]
    # For Model objects
    graph = obj.__dict__.get('_graph', None)
    # Nest methods with graph
    if graph is None: return meth(*args, **kwargs)
    else:
      assert isinstance(graph, tf.Graph)
      with graph.as_default(): return meth(*args, **kwargs)
  return wrapper


def init_with_graph(init):
  def wrapper(*args, **kwargs):
    assert isinstance(tfr.current_graph, tf.Graph)
    with tfr.current_graph.as_default(): init(*args, **kwargs)
  return wrapper
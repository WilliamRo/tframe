import tensorflow as tf


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

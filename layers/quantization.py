from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe.layers.layer import Layer, single_input



class Binarize(Layer):

  full_name = 'binarize'
  abbreviation = 'binarize'

  def __init__(self): pass


  @single_input
  def _link(self, x, **kwargs):
    assert isinstance(x, tf.Tensor)

    @tf.custom_gradient
    def sign(_x):
      def grad(dy): return dy * tf.cast(-1 < _x < 1.0, tf.float32)
      return tf.sign(_x), grad

    return sign(x)



if __name__ == '__main__':
  from tframe import console
  from tframe import tf
  console.suppress_logging()
  tf.InteractiveSession()

  a, b = 1.8, -100.0
  console.eval_show(tf.sign(a))
  console.eval_show(tf.sign(b))

  def grad(x):
    return tf.cast(-1 < x < 1.0, tf.float32)

  console.eval_show(grad(0.9))
  console.eval_show(grad(a))
  console.eval_show(grad(b))


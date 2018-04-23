from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import hub

from tframe.models.model import Model
from tframe.nets import RNet
from tframe.layers import Input

from tframe.core.decorators import with_graph
from tframe.core import TensorSlot


class Recurrent(Model, RNet):
  """Recurrent neural network base class"""
  # TODO: where will this variable be used?
  model_name = 'RNN'

  def __init__(self, mark=None):
    Model.__init__(self, mark)
    RNet.__init__(self, 'RecurrentNet')
    # Attributes
    # mascot will be initiated as a placeholder with no shape specified
    # .. and will be put into initializer argument of tf.scan
    self._mascot = None

  # region : Build

  @with_graph
  def build(self):
    # self.init_state should be called for the first time inside this method
    #  so that it can be initialized within the appropriate graph

    # Do some initialization
    self._mascot = tf.placeholder(dtype=hub.dtype, name='mascot')

    # :: Define output
    # Make sure input has been defined
    if self.input_ is None: raise ValueError('!! input not found')
    assert isinstance(self.input_, Input)
    self.input_.set_group_shape((None, None))
    # Transpose input so as to fit the input of tf.scan
    input_placeholder = self.input_()
    assert isinstance(input_placeholder, tf.Tensor)
    perm = list(range(len(input_placeholder.shape.as_list())))
    elems = tf.transpose(input_placeholder, [1, 0] + perm[2:])
    # Call scan to produce a dynamic op
    scan_outputs, _ = tf.scan(
      self, elems, initializer=(self._mascot, self.init_state), name='Scan')
    # Transpose scan outputs to get final outputs
    assert isinstance(scan_outputs, tf.Tensor)
    perm = list(range(len(scan_outputs.shape.as_list())))
    self.outputs.plug(tf.transpose(scan_outputs, [1, 0] + perm[2:]))

    # Set built flag
    self._built = True

  # endregion: Build

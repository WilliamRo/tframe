from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import config
from tframe import FLAGS

from tframe.models.model import Model
from tframe.nets import RecurrentNet
from tframe.layers import Input

from tframe.core.decorators import with_graph


class Recurrent(Model, RecurrentNet):
  """Recurrent neural network base class"""
  # TODO: where will this variable be used?
  model_name = 'RNN'

  def __init__(self, mark=None):
    Model.__init__(self, mark)
    RecurrentNet.__init__(self, 'RNN')
    # Attributes
    self._input_for_infer = None
    self._prediction = None

    # mascot will be initiated as a placeholder with no shape specified
    # .. and will be put into initializer argument of tf.scan
    self._mascot = None


  # region : Properties

  @property
  def description(self):
    return self.structure_string()

  # endregion : Properties

  # region : Build

  @with_graph
  def build(self, batch_size, num_steps):
    # self.init_state should be called for the first time inside this method
    #  so that it can be initialized within the appropriate graph

    # Do some initialization
    self._mascot = tf.placeholder(dtype=config.dtype, name='mascot')
    self._init_input_for_inference()
    self.set_group_shape(batch_size, num_steps)

    # Core
    # self._init_prediction()
    self._define_loss()

    # Print status and model structure
    self.show_building_info(RecurrentNet=self)

    # Launch model
    self.launch_model(FLAGS.overwrite)

    # Set built flag
    self._built = True

  def _init_input_for_inference(self):
    assert isinstance(self.input_, Input)
    assert self.input_.sample_shape[0] is not None
    self._input_for_infer = tf.placeholder(
      dtype=config.dtype, shape=[None, 1] + self.input_.sample_shape,
      name='inference_input')

  def _init_prediction(self):
    self._prediction, _ = tf.scan(
      self, self._input_for_infer, initializer=(self._mascot, self.init_state),
      back_prop=False, name='Inference')

  def _define_loss(self):
    # Generate element for tf.scan
    input_tensor = self.input_()
    assert isinstance(input_tensor, tf.Tensor)
    perm = list(range(len(input_tensor.shape.as_list())))
    elems = tf.transpose(input_tensor, [1, 0] + perm[2:])

    outputs, final_states = tf.scan(
      self, elems, initializer=(self._mascot, self.init_state), name='Train')

    self._loss = None

  # endregion: Build

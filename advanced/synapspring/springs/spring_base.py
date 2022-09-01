import numpy as np

from collections import OrderedDict

from tframe.core.nomear import Nomear
from tframe import tf
from tframe import hub as th



class SpringBase(Nomear):

  def __init__(self, model):
    # Every spring use shadow vars
    th.create_shadow_vars = True

    # tframe.model
    self._model = model

    # Init omega
    self.omegas = OrderedDict()


  # region: Properties

  @property
  def model(self):
    from tframe.models.sl.predictor import Predictor
    assert isinstance(self._model, Predictor)
    return self._model

  @property
  def variables(self):
    return tf.trainable_variables()

  # endregion: Properties

  # region: Abstract Methods

  def calculate_loss(self) -> tf.Tensor:
    vars = self.model.var_list
    shadows = self.model.shadows
    assert len(vars) == len(shadows)

    loss_list = []
    for v in vars:
      s = shadows[v]
      loss_list.append(tf.reduce_mean(tf.square(s - v)))

    return tf.multiply(th.cl_reg_lambda, tf.add_n(loss_list), name='cl_reg_l2')

  def init_after_linking_before_calc_loss(self):
    # Initialize omega as zeros
    with tf.name_scope('Shadows'):
      for v in self.variables:
        name = v.name.split(':')[0] + '-omega'
        shape = v.shape.as_list()
        shadow = tf.Variable(np.zeros(shape, dtype=np.float32),
                             trainable=False, name=name, shape=shape)
        self.omegas[v] = shadow

  def call_after_each_update(self):
    pass

  def update_omega_after_training(self):
    pass

  # endregion: Abstract Methods



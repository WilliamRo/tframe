import numpy as np

from collections import OrderedDict

from tframe import console
from tframe import context
from tframe.core.nomear import Nomear
from tframe import tf
from tframe import hub as th



class SpringBase(Nomear):

  name = 'CL-REG-L2'

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
      omega = self.omegas[v]
      loss_list.append(tf.reduce_mean(omega * tf.square(s - v)))

    return tf.multiply(th.cl_reg_lambda, tf.add_n(loss_list), name=self.name)

  def init_after_linking_before_calc_loss(self):
    """This method will be called inside Predictor._build"""
    # Initialize omega as zeros
    with tf.name_scope('Omega'):
      for v in self.variables:
        name = v.name.split(':')[0] + '-omega'
        shape = v.shape.as_list()
        shadow = tf.Variable(np.zeros(shape, dtype=np.float32),
                             trainable=False, name=name, shape=shape)
        self.omegas[v] = shadow

    self._show_status('Omegas has been initiated.')

  def call_after_each_update(self):
    pass

  def update_omega_after_training(self):
    self.model.agent.load()

    self._update_omega()

    context.trainer._save_model()
    self._show_status('Omegas has been saved.')

  def _update_omega(self):
    ops = []
    for v in self.variables:
      shape = v.shape.as_list()
      ops.append(tf.assign(self.omegas[v], 1e6 * np.ones(shape)))
    self.model.session.run(ops)

  # endregion: Abstract Methods

  # region: Private Methods

  def _show_status(self, text):
    console.show_status(text, symbol=f'[{self.name}]')

  # endregion: Private Methods



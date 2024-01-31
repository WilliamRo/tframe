from tframe.core import Group
from tframe.core import TensorSlot, OperationSlot
from tframe.layers.common import Input
from tframe.models.model import Model
from tframe.nets.net import Net
from tframe import context
from tframe import DataSet
from tframe import hub as th
from tframe import pedia
from tframe import tf

import numpy as np



class GAN(Model):
  """Generative Adversarial Networks"""

  def __init__(self, mark, G_input_shape, D_input_shape):
    # Call parent's constructor
    super().__init__(mark)

    # Define generator and discriminator
    self.Generator = Net(pedia.Generator)
    self.Discriminator = Net(pedia.Discriminator)
    # Alias
    self.G = self.Generator
    self.D = self.Discriminator

    # Add input layers
    self.G.add(Input(sample_shape=G_input_shape, name=pedia.G_input))
    self.D.add(Input(sample_shape=D_input_shape, name=pedia.D_input))

    # Private fields
    self._G_output = None
    self._Dr, self._Df = None, None
    self._logits_Dr, self._logits_Df = None, None

    self._loss_D = TensorSlot(self, 'Loss-D')
    self._loss_G = TensorSlot(self, 'Loss-G')
    self._loss_Dr, self._loss_Df = None, None

    self._train_step_D = OperationSlot(self, name='Train-step-D')
    self._train_step_G = OperationSlot(self, name='Train-step-G')

    # These two guys will be run in self.update_model method
    self._update_group_D = Group(self, self._loss_D, self._train_step_D,
                                 name='Update-group-D')
    self._update_group_G = Group(self, self._loss_G, self._train_step_G,
                                 name='Update-group-G')

  # region : Properties

  @property
  def description(self):
    return [f'Generator: {self.G.structure_string()}',
            f'Discriminator: {self.D.structure_string()}']

  # endregion : Properties

  # region: Public Methods

  def generate(self, z=None, sample_num=1):
    # Check model and session
    if not self.G.linked: raise AssertionError('!! Model not built yet.')
    if not self.launched: self.launch_model(overwrite=False)

    # Get sample number
    sample_num = (th.sample_num if th.sample_num > 0 else max(1, sample_num))

    # Check input z
    z = self._random_z(sample_num) if z is None else z
    z_shape = list(z.shape[1:])
    g_input_shape = self.G.input_tensor.get_shape().as_list()[1:]
    if z_shape != g_input_shape:
      raise ValueError("!! Shape of input z {} doesn't match the shape of "
                       "generator's input {}".format(z_shape, g_input_shape))

    # Generate samples
    feed_dict = {self.G.input_tensor: z}
    feed_dict.update(self.agent.get_status_feed_dict(is_training=False))
    samples = self.outputs.run(feed_dict)
    return samples

  # endregion: Public Methods

  # region: Private Methods

  def _build(self, optimizer=None, loss=pedia.cross_entropy,
             G_optimizer=None, D_optimizer=None, **kwargs):
    # Link generator
    self._G_output = self.Generator()

    # Make sure G_out can be fed into D
    g_shape = self._G_output.get_shape().as_list()[1:]
    d_shape = self.D.input_.input_shape[1:]
    if g_shape != d_shape:
      raise ValueError('Output shape of generator {} does not match the input '
                       'shape of discriminator {}'.format(g_shape, d_shape))

    # Plug self._G_output to GAN.output slot
    self.outputs.plug(self._G_output)

    # Link discriminator
    logits_dict = context.logits_tensor_dict
    self._Dr = self.Discriminator()
    self._logits_Dr = logits_dict.pop(list(logits_dict.keys())[0])
    self._Df = self.Discriminator(self._G_output)
    self._logits_Df = logits_dict.pop(list(logits_dict.keys())[0])

    # Define loss (extra losses are not supported yet)
    with tf.name_scope('Losses'):
      self._define_losses(loss, kwargs.get('smooth_factor', 0.9))

    # Define train steps
    if G_optimizer is None:
      G_optimizer = tf.train.AdamOptimizer(th.learning_rate)
    if D_optimizer is None:
      D_optimizer = tf.train.AdamOptimizer(th.learning_rate)

    # TODO: self.[DG].parameters should be checked
    with tf.name_scope('Train_Steps'):
      with tf.name_scope('G_train_step'):
        self._train_step_G.plug(G_optimizer.minimize(
          loss=self._loss_G.tensor, var_list=self.G.parameters))
      with tf.name_scope('D_train_step'):
        self._train_step_D.plug(D_optimizer.minimize(
          loss=self._loss_D.tensor, var_list=self.D.parameters))

  def _define_losses(self, loss, alpha):
    """To add extra losses, e.g., regularization losses, this method should be
    overwritten"""
    if callable(loss):
      self._loss_G, self._loss_D = loss(self)
      assert False
      return
    elif not isinstance(loss, str):
      raise TypeError('loss must be callable or a string')

    loss = loss.lower()
    if loss == pedia.default:
      loss_Dr_raw = -tf.log(self._Dr, name='loss_D_real_raw')
      loss_Df_raw = -tf.log(1. - self._Df, name='loss_D_fake_raw')
      loss_G_raw = -tf.log(self._Df, name='loss_G_raw')
    elif loss == pedia.cross_entropy:
      loss_Dr_raw = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self._logits_Dr, labels=tf.ones_like(self._logits_Dr) * alpha)
      loss_Df_raw = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self._logits_Df, labels=tf.zeros_like(self._logits_Df))
      loss_G_raw = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self._logits_Df, labels=tf.ones_like(self._logits_Df))
    else:
      raise ValueError('Can not resolve "{}"'.format(loss))

    with tf.name_scope('D_losses'):
      loss_Dr = tf.reduce_mean(loss_Dr_raw, name='loss_D_real')
      loss_Df = tf.reduce_mean(loss_Df_raw, name='loss_D_fake')
      loss_D = tf.add(loss_Dr, loss_Df, name='loss_D')
      self._loss_D.plug(loss_D)
    with tf.name_scope('G_loss'):
      self._loss_G.plug(tf.reduce_mean(loss_G_raw, name='loss_G'))

  def _random_z(self, sample_num):
    z_dim = self.G.input_tensor.get_shape().as_list()[1]
    z = np.random.standard_normal(size=[sample_num, z_dim])
    return z

  # endregion: Private Methods

  # region: Overwriting

  def update_model(self, data_batch, **kwargs):
    assert isinstance(data_batch, DataSet)

    # (1) Update D
    feed_dict_D = {self.D.input_tensor: data_batch.features,
                   self.G.input_tensor: self._random_z(data_batch.size)}
    feed_dict_D.update(self.agent.get_status_feed_dict(is_training=True))
    results = self._update_group_D.run(feed_dict_D)

    # (2) Update G
    feed_dict_G = {self.G.input_tensor: self._random_z(data_batch.size)}
    feed_dict_G.update(self.agent.get_status_feed_dict(is_training=True))
    results.update(self._update_group_G.run(feed_dict_G))

    return results

  def handle_structure_detail(self):
    G_rows, G_total_params, G_dense_total = self.G.structure_detail
    D_rows, D_total_params, D_dense_total = self.D.structure_detail

    # Take some notes
    params_str = 'Generator total params: {}'.format(G_total_params)
    self.agent.take_notes(params_str)
    params_str = 'Discriminator total params: {}'.format(D_total_params)
    self.agent.take_notes(params_str)

    if th.show_structure_detail:
      print('.. Generator structure detail:\n{}'.format(G_rows))
      print('.. Discriminator structure detail:\n{}'.format(D_rows))

    if th.export_structure_detail:
      self.agent.take_notes('Structure detail of Generator:', False)
      self.agent.take_notes(G_rows, False)
      self.agent.take_notes('Structure detail of Discriminator:', False)
      self.agent.take_notes(D_rows, False)

  # endregion: Overwriting


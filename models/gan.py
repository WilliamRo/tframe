from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from ..models.model import Model
from ..nets.net import Net
from ..layers import Input
from ..utils import imtool

from .. import pedia
from .. import FLAGS

flags = tf.app.flags

flags.DEFINE_bool('fix_sample_z', False, 'Whether to fix z when snapshotting')
flags.DEFINE_integer('sample_num', -1, 'Number of samples to generate')


class GAN(Model):
  """"""
  def __init__(self, z_dim=None, sample_shape=None, output_shape=None,
               mark=None, fix_sample_z=None, sample_num=9):
    # Call parent's constructor
    Model.__init__(self, mark)

    # Define generator and discriminator
    self.Generator = Net(pedia.Generator)
    self.Discriminator = Net(pedia.Discriminator)
    # Alias
    self.G = self.Generator
    self.D = self.Discriminator

    # If z_dim/sample_shape is provided, define the input for
    #   generator/discriminator accordingly
    if z_dim is not None:
      self.G.add(Input(shape=[None, z_dim], name='z'))
    if sample_shape is not None:
      if (not isinstance(sample_shape, list) and
            not isinstance(sample_shape, tuple)):
        raise TypeError('sample shape must be a list or a tuple')
      self.D.add(Input(shape=[None] + list(sample_shape), name='samples'))

    self._z_dim = z_dim
    self._sample_shape = sample_shape
    self._output_shape = output_shape
    self._sample_num = sample_num
    self._fix_sample_z = (fix_sample_z if fix_sample_z is not None
                          else FLAGS.fix_sample_z)

    # Private tensors and ops
    self._G, self._outputs = None, None
    self._Dr, self._Df = None, None
    self._logits_Dr, self._logits_Df = None, None
    self._loss_G, self._loss_D = None, None
    self._loss_Dr, self._loss_Df = None, None
    self._train_step_G, self._train_step_D = None, None
    self._merged_summary_G, self._merged_summary_D = None, None

    self._sample_z = None

  @property
  def _theta_D(self):
    return [var for var in tf.trainable_variables()
             if pedia.Discriminator in var.name]

  @property
  def _theta_G(self):
    return [var for var in tf.trainable_variables()
            if pedia.Generator in var.name]

  def build(self, loss='default', optimizer=None):
    """
    Build model
    :param loss: either a string or a function with:
                  (1) params: an instance of GAN
                  (2) return: G_loss, D_loss
    :param optimizer: a tensorflow optimizer
    """
    # Link G and D to produce _G and _D
    self._G = self.Generator()
    # :: Check shape
    g_shape = self._G.get_shape().as_list()[1:]
    d_shape = self.D.inputs.get_shape().as_list()[1:]
    if g_shape != d_shape:
      raise ValueError('Output shape of generator {} does not match the input '
                        'shape of discriminator {}'.format(g_shape, d_shape))

    self._Dr, self._logits_Dr = self.Discriminator(with_logits=True)
    self._Df, self._logits_Df = self.Discriminator(self._G, with_logits=True)

    # Define output tensor
    if self._output_shape is None:
      self._output_shape = g_shape
    if g_shape == self._output_shape:
      self._outputs = self._G
    else:
      self._outputs = tf.reshape(
        self._G, shape=[-1] + self._output_shape, name='outputs')

    # Prepare sample z
    self._sample_z = tf.Variable(
      initial_value=tf.random_normal(
        shape=[self._sample_num, self.G.inputs.get_shape().as_list()[1]]),
      trainable=False)

    # Define loss
    self._define_losses(loss)

    # Define train steps
    if optimizer is None:
      optimizer = tf.train.AdamOptimizer()
    get_train_step = lambda loss_, var_list: optimizer.minimize(
      loss=loss_, var_list=var_list)
    with tf.name_scope('G_train_step'):
      self._train_step_G = get_train_step(self._loss_G, self._theta_G)
    with tf.name_scope('D_train_step'):
      self._train_step_D = get_train_step(self._loss_D, self._theta_D)

    # Add summaries
    self._add_summaries()

    # Print status and model structure
    self.show_building_info(Generator=self.G, Discriminator=self.D)

    # Set default snapshot function
    self._snapshot_function = self._default_snapshot_function

    # Launch session
    self.launch_model(FLAGS.overwrite)

  def _define_losses(self, loss):
    if callable(loss):
      self._loss_G, self._loss_D = loss(self)
      return
    elif not isinstance(loss, six.string_types):
      raise TypeError('loss must be callable or a string')

    loss = loss.lower()
    if loss == pedia.default:
      with tf.name_scope('D_losses'):
        self._loss_Dr = tf.reduce_mean(-tf.log(self._Dr), name='loss_D_real')
        self._loss_Df = tf.reduce_mean(-tf.log(1. - self._Df),
                                       name='loss_D_fake')
        self._loss_D = tf.add(self._loss_Dr, self._loss_Df, name='loss_D')
      with tf.name_scope('G_loss'):
        self._loss_G = tf.reduce_mean(-tf.log(self._Df), name='loss_G')
    elif loss == pedia.cross_entropy:
      with tf.name_scope('D_losses'):
        self._loss_Dr = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self._logits_Dr, labels=tf.ones_like(self._logits_Dr)),
          name='loss_D_real')
        self._loss_Df = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self._logits_Df, labels=tf.zeros_like(self._logits_Df)),
          name='loss_D_fake')
        self._loss_D = tf.add(self._loss_Dr, self._loss_Df, name='loss_D')
      with tf.name_scope('G_loss'):
        self._loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          logits=self._logits_Df, labels=tf.ones_like(self._logits_Df)))
    else:
      raise ValueError('Can not resolve "{}"'.format(loss))

  def _add_summaries(self):
    sum_z = tf.summary.histogram('z_sum', self.G.inputs)
    sum_Dr = tf.summary.histogram("D_real_sum", self._Dr)
    sum_Df = tf.summary.histogram("D_fake_sum", self._Df)

    sum_loss_G = tf.summary.scalar("G_loss_sum", self._loss_G)
    sum_loss_Dr = tf.summary.scalar("D_real_loss_sum", self._loss_Dr)
    sum_loss_Df = tf.summary.scalar("D_fake_loss_sum", self._loss_Df)
    sum_loss_D = tf.summary.scalar("D_loss_sum", self._loss_D)

    sum_G_list = [sum_loss_G, sum_Df, sum_loss_Df, sum_z]
    if len(self._output_shape) == 3:
      sum_G_list += [tf.summary.image("G_sum", self._outputs, max_outputs=3)]
    self._merged_sum_G = tf.summary.merge(sum_G_list)

    self._merged_sum_D = tf.summary.merge([sum_Dr, sum_loss_Dr, sum_loss_D])

  def _update_model(self, data_batch, **kwargs):
    # TODO: design some mechanisms to handle these
    G_times = kwargs.get('G_times', 1)
    D_times = kwargs.get('D_times', 1)

    features = data_batch[pedia.features]
    sample_num = features.shape[0]

    loss_D, loss_G = None, None
    summaries_D, summaries_G = None, None

    assert isinstance(self._session, tf.Session)
    # Update discriminator
    feed_dict_D = {self.D.inputs: features,
                   self.G.inputs: self._random_z(sample_num)}
    feed_dict_D.update(self._get_status_feed_dict(is_training=True))

    for _ in range(D_times):
      _, loss_D, summaries_D = self._session.run(
        [self._train_step_D, self._loss_D, self._merged_sum_D],
        feed_dict=feed_dict_D)

    # Update generator
    feed_dict_G = {self.G.inputs: self._random_z(sample_num)}
    feed_dict_G.update(self._get_status_feed_dict(is_training=True))

    for _ in range(G_times):
      _, loss_G, summaries_G = self._session.run(
        [self._train_step_G, self._loss_G, self._merged_sum_G],
        feed_dict=feed_dict_G)

    # Write summaries to file
    assert isinstance(self._summary_writer, tf.summary.FileWriter)
    self._summary_writer.add_summary(summaries_D, self._counter)
    self._summary_writer.add_summary(summaries_G, self._counter)

    # Return loss dict
    return {'Discriminator loss': loss_D, 'Generator loss': loss_G}

  def _random_z(self, sample_num):
    assert isinstance(self.G.inputs, tf.Tensor)
    z_dim = self.G.inputs.get_shape().as_list()[1]
    return np.random.standard_normal(size=[sample_num, z_dim])

  @staticmethod
  def _default_snapshot_function(self):
    # Generate samples
    feed_dict = {self.G.inputs: (self._sample_z.eval() if self._fix_sample_z
                                 else self._random_z(self._sample_num))}
    feed_dict.update(self._get_status_feed_dict(is_training=False))
    samples = self._outputs.eval(feed_dict)

    # Plot samples
    fig = imtool.gan_grid_plot(samples)

    return fig

  def generate(self, z=None, sample_num=1):
    if self._G is None:
      raise ValueError('Model not built yet')

    if self._session is None:
      self.launch_model(overwrite=False)
    assert isinstance(self._session, tf.Session)

    # Get sample number
    sample_num = (FLAGS.sample_num if FLAGS.sample_num > 0 else
                  max(1, sample_num))

    # Check input z
    z = self._random_z(sample_num) if z is None else z
    z_shape = list(z.shape[1:])
    g_input_shape = self.G.inputs.get_shape().as_list()[1:]
    if z_shape != g_input_shape:
      raise ValueError("Shape of input z {} doesn't match the shape of "
                        "generator's input {}".format(
        z_shape, g_input_shape))

    # Generate samples
    feed_dict = {self.G.inputs: z}
    feed_dict.update(self._get_status_feed_dict(is_training=False))
    samples = self._session.run(self._outputs, feed_dict=feed_dict)

    return samples









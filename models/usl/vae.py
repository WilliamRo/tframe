from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tframe.models.model import Model
from tframe.nets.net import Net
from tframe.layers import Input

from tframe.utils import imtool

from tframe import pedia
from tframe import hub
from tframe import DataSet
from tframe.core import with_graph


class VAE(Model):
  """Model for Variational Autoencoders"""
  def __init__(self, z_dim=None, sample_shape=None, output_shape=None,
              mark=None, classes=0):
    # Call parent's constructor
    Model.__init__(self, mark)

    # Fields
    self._output_shape = output_shape

    # Define encoder and decoder
    self.Encoder = Net(pedia.Encoder)
    self.Decoder = Net(pedia.Decoder)

    self.Q = self.Encoder
    self.P = self.Decoder

    # If z_dim/sample_shape is provided, define the input for
    #   decoder/encoder accordingly
    if z_dim is not None:
      self.P.add(Input(sample_shape=[None, z_dim], name='z'))
    if sample_shape is not None:
      if (not isinstance(sample_shape, list) and
           not isinstance(sample_shape, tuple)):
        raise TypeError('sample shape must be a list or a tuple')
      self.Q.add(Input(
        sample_shape=[None] + list(sample_shape), name='samples'))

    # Placeholders
    self._sample_num = None
    self._classes = classes
    self._conditional = classes > 0
    if self._conditional:
      self._targets = tf.placeholder(
        dtype=tf.float32, shape=[None, classes], name='one_hot_labels')

    self._P, self._outputs = None, None

    # ...
    pass


  # region : Properties

  @property
  def description(self):
    str = 'Encoder: {}\nDecoder: {}\n'.format(
      self.Q.structure_string(), self.P.structure_string())
    return str

  # endregion : Properties

  # region : Building

  @with_graph
  def _build(self, optimizer=None):
    # Generate mean and var from encoder
    z_mu, z_logvar = self.Encoder()
    # Sample z~N(mu, var)
    with tf.name_scope('sample_z'):
      eps = tf.random_normal(shape=tf.shape(z_mu))
      z_sample = z_mu + tf.exp(z_logvar / 2) * eps
    # Link decoder for training and output
    _, logits = self.Decoder(z_sample, with_logits=True)
    self._P = self.Decoder()
    # Check output shape of decoder
    p_shape = self._P.get_shape().as_list()[1:]
    q_shape = self.Q.input_tensor.get_shape()[1:]
    if p_shape != q_shape:
      raise ValueError('Output shape of decoder {} does not match the input '
                        'shape of encoder {}'.format(p_shape, q_shape))
    # Define the output tensor
    if self._output_shape is None:
      self._output_shape = p_shape
    if p_shape == self._output_shape:
      self._outputs = self._P
    else:
      self._outputs = tf.reshape(
        self._P, shape=[-1] + self._output_shape, name='outputs')

    # Define loss
    with tf.name_scope('Losses'):
      # E[log P(X|z)]
      recon_loss = tf.reduce_mean(tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits, labels=self.Q.input_tensor), 1))
      # recon_loss = tf.norm

      # D_KL(Q(z|X) || P(z|X))
      kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(
        tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1))
      # VAE loss
      vae_loss = recon_loss + kl_loss
      self._loss = vae_loss

    # Add summaries
    with tf.name_scope('Summaries'):
      self._merged_summary = tf.summary.merge([
        tf.summary.scalar('recon_loss', recon_loss),
        tf.summary.scalar('kl_loss', kl_loss),
        tf.summary.scalar('vae_loss', vae_loss)])

    # Check optimizer
    if optimizer is None:
      optimizer = tf.train.AdamOptimizer()

    # Define training step
    with tf.name_scope('Train_Step'):
      self._train_step = optimizer.minimize(vae_loss)

    # Print status and model structure
    self._show_building_info(Encoder=self.Q, Decoder=self.P)

    # Set default snapshot function TODO
    self._snapshot_function = self._default_snapshot_function

    # Launch session
    self.launch_model(overwrite=hub.overwrite)

  # endregion : Building

  # region : Training

  def pretrain(self, **kwargs):
    self._sample_num = (hub.sample_num if hub.sample_num > 0
                        else kwargs.get('sample_num', 9))

  def update_model(self, data_batch, **kwargs):
    assert isinstance(data_batch, DataSet)
    features = data_batch[pedia.features]

    assert isinstance(self._session, tf.Session)

    # Set feed dictionary
    feed_dict = {self.Q.input_tensor: features}
    feed_dict.update(self._get_status_feed_dict(is_training=True))
    _, loss, summaries = self._session.run(
      [self._train_step, self._loss, self._merged_summary], feed_dict=feed_dict)

    # Write summaries to file
    assert isinstance(self._summary_writer, tf.summary.FileWriter)
    self._summary_writer.add_summary(summaries, self.counter)

    # Return loss dict
    return {'VAE loss': loss}

  # endregion : Training

  # region : Private Methods

  @staticmethod
  def _default_snapshot_function(self):
    assert isinstance(self, VAE)

    z = self._random_z(self._sample_num)

    feed_dict = {}
    feed_dict[self.P.input_tensor] = z
    feed_dict.update(self._get_status_feed_dict(is_training=False))
    samples = self._outputs.eval(feed_dict)

    # Plot samples
    fig = imtool.gan_grid_plot(samples)

    return fig

  def _random_z(self, sample_num, with_label=False):
    assert isinstance(self.P.input_tensor, tf.Tensor)
    z_dim = self.P.input_tensor.get_shape().as_list()[1]
    z = np.random.standard_normal(size=[sample_num, z_dim])

    # if self._conditional and with_label:
    #   labels = self._random_labels(sample_num)
    #   return z, labels

    return z

  # endregion : Private Methods

  # region : Public Methods

  def generate(self, z=None, sample_num=1, labels=None):
    """
    Generate samples with the same specification with that in GAN
    :param z: 
    :param sample_num: 
    :param labels: 
    :return: 
    """
    # Check model and session
    if self._P is None:
      raise ValueError('Model not built yet')
    if self._session is None:
      self.launch_model(overwrite=False)
    assert isinstance(self._session, tf.Session)

    # Get sample numbers
    sample_num = (hub.sample_num if hub.sample_num > 0 else
                  max(1, sample_num))

    # Check input z
    z = self._random_z(sample_num) if z is None else z
    sample_num = z.shape[0]
    z_shape = list(z.shape[1:])
    p_input_shape = self.P.input_tensor.get_shape().as_list()[1:]
    if z_shape != p_input_shape:
      raise ValueError("Shape of input z {} doesn't match the shape of "
                       "decoder's input {}".format(z_shape, p_input_shape))

    # Generate samples
    feed_dict = {self.P.input_tensor: z}
    feed_dict.update(self._get_status_feed_dict(is_training=False))
    samples = self._session.run(self._outputs, feed_dict=feed_dict)

    return samples

  # endregion : Public Methods

  """For some reason, do not delete this line"""


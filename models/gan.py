from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from ..models.model import Model
from ..nets.net import Net
from ..layers import Input

from .. import console
from .. import pedia
from .. import FLAGS


class GAN(Model):
  """"""
  def __init__(self, z_dim=None, sample_shape=None, mark=None):
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

    # Private tensors and ops
    self._G = None
    self._Dr, self._Df = None, None
    self._logits_Dr, self._logits_Df = None, None
    self._loss_G, self._loss_D = None, None
    self._loss_Dr, self._loss_Df = None, None
    self._train_step_G, self._train_step_D = None, None
    self._merged_summary_G, self._merged_summary_D = None, None

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

    # Define loss
    self._define_losses(loss)

    # Define train steps
    if optimizer is None:
      optimizer = tf.train.AdamOptimizer()
    get_train_step = lambda loss, var_list: optimizer.minimize(
      loss=loss, var_list=var_list)
    with tf.name_scope('G_train_step'):
      self._train_step_G = get_train_step(self._loss_G, self._theta_G)
    with tf.name_scope('D_train_step'):
      self._train_step_D = get_train_step(self._loss_D, self._theta_D)

    # Add summaries
    self._add_summaries()

    # Print status and model structure
    self.show_building_info(Generator=self.G, Discriminator=self.D)

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
          logits=self._logits_Df, labels=tf.ones_like(self._logits_Dr)))
    else:
      raise ValueError('Can not resolve "{}"'.format(loss))

  def _add_summaries(self):
    sum_Dr = tf.summary.histogram("D_real_sum", self._Dr)
    sum_Df = tf.summary.histogram("D_fake_sum", self._Df)

    sum_loss_G = tf.summary.scalar("G_loss_sum", self._loss_G)
    sum_loss_Dr = tf.summary.scalar("D_real_loss_sum", self._loss_Dr)
    sum_loss_Df = tf.summary.scalar("D_fake_loss_sum", self._loss_Df)
    sum_loss_D = tf.summary.scalar("D_loss_sum", self._loss_D)

    self._merged_sum_G = tf.summary.merge([sum_loss_G, sum_Df, sum_loss_Df])
    self._merged_sum_D = tf.summary.merge([sum_Dr, sum_loss_Dr, sum_loss_D])





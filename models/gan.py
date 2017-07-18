from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..models.model import Model
from ..nets.net import Net
from ..layers import Input

from .. import FLAGS


class GAN(Model):
  """"""
  def __init__(self, z_dim=None, sample_shape=None, mark=None):
    # Call parent's constructor
    Model.__init__(self, mark)

    # Define generator and discriminator
    self.Generator = Net('Generator')
    self.Discriminator = Net('Discriminator')
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
    self._loss_G, self._loss_G = None, None
    self._loss_Dr, self._loss_Df = None, None
    self._train_step_G, self._train_step_D = None, None
    self._merged_summary_G, self._merged_summary_D = None, None

  def build(self, loss='default', optimizer=None):
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

    # Launch session
    self.launch_model(FLAGS.overwrite)

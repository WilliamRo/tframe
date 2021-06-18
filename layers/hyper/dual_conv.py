from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker
from tframe import hub as th
from tframe.layers.layer import single_input
from tframe.layers.normalization import BatchNormalization

from .hyper_base import HyperBase


class DualConv2D(HyperBase):

  abbreviation = 'duconv2d'

  def __init__(self,
               filter_generator=None,
               **kwargs):

    # Call parent's initializer
    super(DualConv2D, self).__init__(use_bias=False, **kwargs)


  @single_input
  def _link(self, x: tf.Tensor, **kwargs):

    return None

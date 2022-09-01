from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
from tframe import tf

from tframe import checker
from tframe import hub
from tframe.operators.masked_weights import MaskedWeights


class EtchKernel(MaskedWeights):
  """An EtchKernel is a MaskedWeights initiated with a dense mask."""

  def __init__(self, weights):
    # Created a dense mask based on weights
    assert isinstance(weights, tf.Variable)
    mask = tf.get_variable(
      'etched_surface', shape=weights.shape, dtype=hub.dtype,
      initializer=tf.initializers.ones, trainable=False)
    self.total_size = int(np.prod(weights.shape.as_list()))

    # Call parent's constructor
    super().__init__(weights, mask)

    # Create a mask placeholder for future mask assigning
    self.mask_placeholder = tf.placeholder(
      dtype=hub.dtype, shape=mask.shape, name='mask_placeholder')
    self.assign_mask_op = tf.assign(self.mask, self.mask_placeholder)

    # weights fraction will be assigned during
    #  launching session => model.handle_structure_detail
    #  => net.structure_detail => pruner.get_variable_sizes
    self.weights_fraction = None


  def get_assign_mask_op_dict(self, mask_value):
    assert isinstance(mask_value, np.ndarray)
    assert mask_value.shape == self.mask.shape
    return self.assign_mask_op, {self.mask_placeholder: mask_value}


  def get_etch_op_dict(self):
    """Before this method is called, corresponding buffers should be filled.
       Since each EtchKernel may require different buffers to be set, it is
       not necessary to check buffer here.
    """
    mask = self._get_new_mask()
    assert isinstance(mask, np.ndarray)
    # Update weights_fraction
    self.weights_fraction = 100. * np.sum(mask) / mask.size
    # Clear buffer
    self.clear_buffers()
    return self.get_assign_mask_op_dict(mask)


  def _get_new_mask(self):
    raise NotImplementedError


  @staticmethod
  def get_etch_kernel(kernel_string):
    """kernel string example: `lottery:prune_frac=0.2`
    """
    assert isinstance(kernel_string, str)
    assert re.fullmatch(r'\w+(:\w+=((\d+\.\d*)|\w)(,\w+=((\d+\.\d*)|\w+))*)?',
                        kernel_string) is not None
    if ':' in kernel_string:
      configs = {}
      key, cfg_str = kernel_string.split(':')
      assert isinstance(cfg_str, str)
      for cfg in cfg_str.split(','):
        k, v = cfg.split('=')
        configs[k] = checker.try_str2float(v)
    else: key, configs = kernel_string, {}

    key = key.lower()
    if key == 'cola':
      from tframe.advanced.prune.etches.cola import Cola as Kernel
    elif key == 'lottery':
      from tframe.advanced.prune.etches.lottery import Lottery as Kernel
    elif key == 'eraser':
      from tframe.advanced.prune.etches.eraser import Eraser as Kernel
    else: raise ValueError('!! Unknown etch Kernel `{}`'.format(key))

    return lambda weights: Kernel(weights, **configs)

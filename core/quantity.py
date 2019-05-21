from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tframe as tfr


class Quantity(object):
  """Quantities mainly includes loss and metric defined for a model.
     Quantities are usually calculated as statistical results over
     a batch or the whole data set.
     Quantities are bound with TensorSlots"""

  tf2np = {
    tf.reduce_mean: np.mean,
    tf.norm: np.linalg.norm,
  }

  def __init__(self, kernel, tf_summ_method=None, np_summ_method=None,
               last_only=False, name='Unknown', use_logits=False, **kwargs):
    self._kernel = tfr.checker.check_callable(kernel)
    self._tf_summ_method = tf_summ_method
    if tf_summ_method is not None: tfr.checker.check_callable(tf_summ_method)
    self._np_summ_method = np_summ_method
    if np_summ_method is not None: tfr.checker.check_callable(np_summ_method)

    self._last_only = tfr.checker.check_type(last_only, bool)
    self._quantities = None
    self._quantity = None

    self._use_logits = tfr.checker.check_type(use_logits, bool)
    self._kwargs = kwargs

    self.name = name
    self.lower_is_better = kwargs.get('lower_is_better', True)

  @property
  def support_batch_eval(self):
    self._check_link()
    return self._quantities is not None

  @property
  def quantity(self):
    self._check_link()
    return self._quantity

  @property
  def quantities(self):
    if self._quantities is None:
      self._check_link()
      raise ValueError('!! This quantity does not support `foreach` logic')
    return self._quantities

  @property
  def np_summ_method(self):
    # Check support
    _ = self._quantities
    if self._np_summ_method is not None: return self._np_summ_method
    if self._tf_summ_method not in self.tf2np.keys():
      raise ValueError('!! tf method {} has not been registered'.format(
        self._tf_summ_method))
    self._np_summ_method = self.tf2np[self._tf_summ_method]
    return self.np_summ_method

  def __call__(self, truth, output, **kwargs):
    assert isinstance(truth, tf.Tensor) and isinstance(output, tf.Tensor)
    # Replace output with logits if necessary
    # logits will be registered when softmax layer is being linked
    logits = tfr.context.logits_tensor
    if self._use_logits and output is not logits:
      if logits is None: tfr.console.warning(
        'Logits are supposed to be used in calculating {} '
        'but not found.'.format(self.name))
      else:
        tfr.console.show_status(
          'Logits are used in calculating {}'.format(self.name))
        output = logits

    q = self._kernel(truth, output, **kwargs)
    assert isinstance(q, tf.Tensor)
    # If q is a scalar
    if len(q.shape) == 0:
      self._quantity = q
      if self._tf_summ_method is not None:
        raise AssertionError('!! tf_summ_method is not provided but not used')
      return q

    # Extract result in last time step for RNN output
    if self._last_only:
      assert len(q.shape) > 1
      q = q[:, -1]
    self._quantities = q
    if self._tf_summ_method is None:
      raise TypeError('!! summ_method should be provided')
    self._quantity = self._tf_summ_method(self._quantities)
    assert isinstance(self._quantity, tf.Tensor)
    assert len(self._quantity.shape) == 0
    return self._quantity

  def function(self, truth, output, **kwargs):
    """This method is designed for calculating loss in while-loop for
       RNN models"""
    assert isinstance(truth, tf.Tensor) and isinstance(output, tf.Tensor)
    q = self._kernel(truth, output, **kwargs)
    assert isinstance(q, tf.Tensor)
    # If q is a scalar, return directly
    if len(q.shape) == 0: return q

    # Extract result in last time step for RNN output
    if self._last_only:
      # In single step calculation, q does not have num_step dimension
      assert len(q.shape) > 0
      q = q[-1]
    if self._tf_summ_method is None:
      raise TypeError('!! summ_method should be provided')
    q = self._tf_summ_method(q)
    assert isinstance(q, tf.Tensor) and len(q.shape) == 0
    return q

  def _raise_not_linked_error(self):
    raise ValueError('!! This quantity has not been linked yet')

  def _check_link(self):
    if self._quantity is None: self._raise_not_linked_error()

  def apply_np_summ_method(self, quantities):
    """Used only in model.validate_model method"""
    if not self._last_only:
      tfr.checker.check_type(quantities, np.ndarray)

    # Concatenate if necessary
    if isinstance(quantities, list):
      # Extract data in last step if necessary
      # TODO: this has been done in __call__, check again
      # if self._last_only: quantities = [s[:, -1] for s in quantities]
      if isinstance(quantities[0], np.ndarray):
        quantities = np.concatenate(quantities, axis=0)

    return self.np_summ_method(quantities)




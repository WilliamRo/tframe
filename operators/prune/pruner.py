from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import tensorflow as tf

import tframe as tfr
from tframe.core import VariableSlot

from tframe.operators.prune.etches.etch_kernel import EtchKernel


class Pruner(object):
  """This class is originally designed for finding lottery."""

  def __init__(self, model):
    self._model = model
    self._dense_kernels = []
    self._conv_kernels = []

    # key: tf.Variable, value: EtchKernel
    self.variable_dict = OrderedDict()

    # Show status
    tfr.console.show_status('Pruner created.')

  # region : Properties

  @property
  def th(self): return tfr.hub

  @property
  def total_size(self):
    """Total size of weights to be pruned"""
    return np.sum([knl.total_size for knl in self._dense_kernels])

  @property
  def sparse_size(self):
    """Total size of weights to be pruned"""
    return np.sum([knl.total_size * knl.weights_fraction / 100.
                   for knl in self._dense_kernels])
  @property
  def weights_fraction(self):
    return self.sparse_size / self.total_size * 100

  # endregion : Properties

  # region : Public Methods

  def get_variable_sizes(self, variable):
    """This method is only used by stark"""
    assert variable in self.variable_dict.keys()
    kernel = self.variable_dict[variable]
    assert isinstance(kernel, EtchKernel)
    if kernel.mask_buffer is None:
      self._write_weight_and_mask_buffer('[Stark]')
    mask = kernel.mask_buffer
    assert isinstance(mask, np.ndarray)
    # Return size and total size
    size, total_size = int(np.sum(mask)), mask.size
    # Assign weights_fraction (IMPORTANT)
    kernel.weights_fraction = 100.0 * size / total_size
    return size, total_size

  def register_to_dense(self, weights, etch_config):
    """This method will be called in KernelBase._get_weights"""
    # In some case such as building in RNN, a same weights may be called twice
    # .. e.g. build_while_free -> _link
    if weights in self.variable_dict:
      kernel = self.variable_dict[weights]
      assert isinstance(kernel, EtchKernel)
      return kernel.masked_weights

    kernel_constructor = EtchKernel.get_etch_kernel(etch_config)
    kernel = kernel_constructor(weights)
    self.variable_dict[weights] = kernel
    self._dense_kernels.append(kernel)
    return kernel.masked_weights

  def clear(self):
    pass

  def etch_all(self, prompt='[Etch]'):
    """This method will only be called during training in Trainer._etch"""
    # Take down frac before pruning
    prev_frac = self.weights_fraction
    # etching ia based on weights and mask buffers, which should be fetched
    # here
    self._write_weight_and_mask_buffer()
    # Create empty op list and feed_dict
    ops, feed_dict = [], {}
    for knl in self._dense_kernels:
      assert isinstance(knl, EtchKernel)
      op, d = knl.get_etch_op_dict()
      ops.append(op)
      feed_dict.update(d)
    # Run session to update kernels
    self._update_kernels(ops, feed_dict)
    # Show prune result
    curr_frac = self.weights_fraction
    tfr.console.show_status(
      'Weights fraction decreased from {:.2f} to {:.2f}'.format(
        prev_frac, curr_frac), prompt)

  @staticmethod
  def extractor(*args):
    if not any([tfr.hub.prune_on, tfr.hub.weights_mask_on,
                tfr.hub.export_weights]): return
    pruner = tfr.context.pruner
    if pruner is None:
      tfr.console.warning_with_pause(
        'Pruner.extractor has been called yet prune is not on')
      return
    for kernel in pruner._dense_kernels:
      assert isinstance(kernel, EtchKernel)
      export_dict = tfr.context.variables_to_export
      if tfr.hub.prune_on:
        export_dict[kernel.weight_key] = kernel.weights
        export_dict[kernel.mask_key] = kernel.mask
      elif tfr.hub.weights_mask_on:
        # TODO: this branch is to be deprecated
        export_dict[kernel.weight_key] = kernel.masked_weights

  # endregion : Public Methods

  # region : Private Methods

  def _run_op(self, ops, feed_dict=None):
    from tframe.models.model import Model
    model = self._model
    assert isinstance(model, Model)
    return model.session.run(ops, feed_dict)

  def _write_weight_and_mask_buffer(self, prompt='[Etch]'):
    """Write weight and mask buffer to each etch kernel"""
    fetches = [(knl.weights, knl.mask) for knl in self._dense_kernels]
    buffers = self._run_op(fetches)
    for (w_buffer, m_buffer), knl in zip(buffers, self._dense_kernels):
      assert isinstance(knl, EtchKernel)
      knl.weights_buffer = w_buffer
      knl.mask_buffer = m_buffer
    tfr.console.show_status('Weights and mask buffers fetched.', prompt)

  def _update_kernels(self, ops, feed_dict):
    assert isinstance(ops, list) and isinstance(feed_dict, dict)
    self._run_op(ops, feed_dict)

  # endregion : Private Methods

  # region : Lottery 2018

  def set_init_val_lottery18(self):
    """This method will be called in Agent.launch_model.
       It is the reset part of iterative pruning with resetting
    """
    self._model.session.run([ws.assign_init for ws in self._dense_kernels])

  def prune_and_save_lottery18(self):
    """This method should be called only in the end of trainer.train
       Reference:
       [1] Frankle, etc. THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE,
           TRAINABLE NEURAL NETWORKS. 2018
    """
    # pruning should start from best model if save_model is on
    if tfr.hub.save_model:
      tfr.console.show_status('Loading best model to prune ...')
      self._model.agent.load()

    # Do iterative pruning with resetting
    self.etch_all('[Lottery]')

    # Reset weights
    self._run_op([ws.reset_weights for ws in self._dense_kernels])
    # Show status
    tfr.console.show_status('Weights reset.', '[Lottery]')
    # Save next model
    self.save_next_model_lottery18()

  def save_next_model_lottery18(self):
    from tframe.models.model import Model
    model = self._model
    assert isinstance(model, Model)
    # Reset metrics
    model.metrics_manager.reset_records()
    # Reset counter
    model.counter = 0
    # Update mark
    iteration = self.th.pruning_iterations
    if iteration < 10: k = -1
    else:
      assert iteration >= 10
      k = -2
    assert int(model.mark[k:]) == iteration
    model.mark = model.mark[:k] + str(iteration + 1)
    model.agent.save_model()
    # Show status
    tfr.console.show_status('Model {} saved.'.format(model.mark))

  # endregion : Lottery 2018

  # TODO ======================================================================

  # def _fetch_masked_weights(self):
  #   masked_weights = [ws.masked_weights for ws in self._dense_kernels]
  #   value_masked_weights = self._run_op(masked_weights)
  #   for vmw, ws in zip(value_masked_weights, self._dense_kernels):
  #     assert isinstance(ws, WeightSlot)
  #     ws.masked_weights_buffer = vmw
  #
  # def _set_weights_and_masks(self, p):
  #   assert 0 < p < 1
  #   reset_weights_ops = [ws.reset_weights for ws in self._dense_kernels]
  #   # get_assign_mask_op does the pruning work
  #   set_mask_ops = [
  #     ws.get_assign_mask_op_lottery(p) for ws in self._dense_kernels]
  #   self._run_op([reset_weights_ops, set_mask_ops])
  #   # Show status
  #   tfr.console.show_status('Weights reset and masks updated.')
  #
  # def _fetch_masks(self):
  #   fetches = [ws.mask for ws in self._dense_kernels]
  #   val_masks = self._run_op(fetches)
  #   # Distribute to weight_slots
  #   for slot, val_mask in zip(self._dense_kernels, val_masks):
  #     assert isinstance(slot, WeightSlot)
  #     slot.mask_buffer = val_mask


# class WeightSlot(object):
#   """Weight Slot is not a tframe Slot. Currently this class is a little bit
#      of chaos. It's used for
#      (1) lottery
#      (2) etch
#      (3) generic masked weights
#   """
#
#   def __init__(self, weights, frac=None, mask=None):
#     # Sanity check
#     assert isinstance(weights, tf.Variable)
#     if frac is not None: assert np.isreal(frac) and 0 <= frac <= 1
#     if mask is not None: assert isinstance(mask, (tf.Variable, tf.Tensor))
#
#     self.weights = weights
#     self.init_val = tf.Variable(
#       tf.zeros_like(weights), trainable=False, name='init_val')
#
#     if mask is not None: self.mask = mask
#     else: self.mask = tf.Variable(
#       tf.ones_like(weights), trainable=False, name='mask')
#
#     self.masked_weights = tf.multiply(self.weights, self.mask)
#     self.frac = frac
#
#     # Define assign ops
#     self.assign_init = tf.assign(self.init_val, self.weights)
#     self.reset_weights = tf.assign(self.weights, self.init_val)
#
#     # Define weights and mask placeholder
#     self.masked_weights_buffer = None
#     self.mask_buffer = None
#     # weights fraction will be assigned during
#     #  launching session => model.handle_structure_detail
#     #  => net.structure_detail => pruner.get_variable_sizes
#     self.weights_fraction = None
#
#   @property
#   def scope_abbr(self):
#     """In tframe, RNN model, weight's name may be ../gdu/net_u/W"""
#     scopes = self.weights.name.split('/')
#     return '/'.join(scopes[-3:-1])
#
#   @property
#   def weight_key(self):
#     """Used in Pruner.extractor"""
#     scopes = self.weights.name.split('/')
#     key = '/'.join(scopes[-3:])
#     key = key.split(':')[0]
#     return key
#
#   @property
#   def mask_key(self):
#     return self.weight_key + '_mask'
#
#   def get_assign_mask_op_lottery(self, p):
#     """Weights pruned in previous iterations will be be kept pruned."""
#     assert 0 < p < 1
#     p = p * self.frac
#     w = self.masked_weights_buffer
#     assert isinstance(w, np.ndarray) and np.isreal(self.weights_fraction)
#     assert 0 < self.weights_fraction <= 100
#     weight_fraction = np.ceil(self.weights_fraction * (1 - p))
#     # Get weights magnitude
#     w = np.abs(w)
#     # Create mask
#     mask = np.zeros_like(w, dtype=np.float32)
#     mask[w > np.percentile(w, 100 - weight_fraction)] = 1.0
#     # Return mask assign operator
#     op = tf.assign(self.mask, mask)
#     return op
#
#
#


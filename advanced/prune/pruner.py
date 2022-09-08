from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np

import tframe as tfr
from tframe.enums import SaveMode

from tframe.advanced.prune.etches.etch_kernel import EtchKernel


class Pruner(object):
  """This class is originally designed for finding lottery."""

  def __init__(self, model):
    self._model = model
    self._kernels_to_prune = []
    # self._conv_kernels = []

    # key: tf.Variable, value: EtchKernel
    self.variable_dict = OrderedDict()

    # Show status
    tfr.console.show_status('Pruner created.')

  # region : Properties

  @property
  def weights_list(self):
    return [k.weights for k in self._kernels_to_prune]

  @property
  def th(self): return tfr.hub

  @property
  def total_size(self):
    """Total size of weights to be pruned"""
    return np.sum([knl.total_size for knl in self._kernels_to_prune])

  @property
  def sparse_size(self):
    """Total size of weights to be pruned"""
    return np.sum([knl.total_size * knl.weights_fraction / 100.
                   for knl in self._kernels_to_prune])
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

  def register_to_kernels(self, weights, etch_config):
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
    self._kernels_to_prune.append(kernel)
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
    for knl in self._kernels_to_prune:
      assert isinstance(knl, EtchKernel)
      op, d = knl.get_etch_op_dict()
      ops.append(op)
      feed_dict.update(d)
    # Run session to update kernels
    self._update_kernels(ops, feed_dict)
    # Show prune result
    curr_frac = self.weights_fraction
    if abs(curr_frac - prev_frac) > 0.006:
      tfr.console.show_status(
        'Weights fraction decreased from {:.2f} to {:.2f}'.format(
          prev_frac, curr_frac), prompt)

  @staticmethod
  def extractor(*args):
    if not any([tfr.hub.prune_on, tfr.hub.weights_mask_on, tfr.hub.etch_on,
                tfr.hub.export_weights]): return
    pruner = tfr.context.pruner
    if pruner is None:
      tfr.console.warning_with_pause(
        'Pruner.extractor has been called yet prune is not on')
      return
    for kernel in pruner._kernels_to_prune:
      assert isinstance(kernel, EtchKernel)
      export_dict = tfr.context.variables_to_export
      if tfr.hub.prune_on or tfr.hub.etch_on:
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
    fetches = [(knl.weights, knl.mask) for knl in self._kernels_to_prune]
    buffers = self._run_op(fetches)
    for (w_buffer, m_buffer), knl in zip(buffers, self._kernels_to_prune):
      assert isinstance(knl, EtchKernel)
      knl.weights_buffer = w_buffer
      knl.mask_buffer = m_buffer
    if not tfr.hub.etch_quietly:
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
    self._model.session.run([ws.assign_init for ws in self._kernels_to_prune])

  def prune_and_save_lottery18(self):
    """This method should be called only in the end of trainer.train
       Reference:
       [1] Frankle, etc. THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE,
           TRAINABLE NEURAL NETWORKS. 2018
    """
    if tfr.hub.forbid_lottery_saving or tfr.hub.pruning_rate <= 0: return
    # pruning should start from best model if save_model is on
    if tfr.hub.save_model and tfr.hub.save_mode == SaveMode.ON_RECORD:
      tfr.console.show_status('Loading best model to prune ...')
      self._model.agent.load()

    # Do iterative pruning with resetting
    self.etch_all('[Lottery]')

    # Reset weights
    self._run_op([ws.reset_weights for ws in self._kernels_to_prune])
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

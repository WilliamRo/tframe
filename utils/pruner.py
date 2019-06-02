from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import tensorflow as tf

import tframe as tfr
from tframe.core import VariableSlot


class Pruner(object):

  def __init__(self, model):
    self._model = model
    self._dense_weights = []
    self._conv_filters = []
    self.variable_dict = OrderedDict()

    # Plug in fractions
    self._dense_fraction = VariableSlot(self._model)
    with self._model.graph.as_default():
      self._dense_fraction.plug(tf.Variable(
        initial_value=-1.0, trainable=False, name='dense_fraction'))

    # Show status
    tfr.console.show_status('Pruner created.')

  # region : Properties

  @property
  def th(self): return tfr.hub

  @property
  def dense_fraction(self):
    """Here dense fraction does not necessarily be equal to model weight
       fraction since pruning rates of different components may vary.
    """
    return self._dense_fraction.fetch()

  @dense_fraction.setter
  def dense_fraction(self, value):
    self._dense_fraction.assign(value)

  # endregion : Properties

  # region : Public Methods

  def get_variable_sizes(self, variable):
    assert variable in self.variable_dict.keys()
    slot = self.variable_dict[variable]
    assert isinstance(slot, WeightSlot)
    if slot.value_mask is None: self._fetch_masks()
    mask = slot.value_mask
    assert isinstance(mask, np.ndarray)
    # Return size and total size
    size, total_size = int(np.sum(mask)), mask.size
    # Assign weights_fraction (IMPORTANT)
    slot.weights_fraction = 100.0 * size / total_size
    return size, total_size

  def set_init_val(self):
    # Set dense fraction
    self.dense_fraction = 100.0
    # Set init_vals
    self._model.session.run([ws.assign_init for ws in self._dense_weights])

  def register_with_mask(self, weights, mask):
    """This method will be called only by linker.get_masked_weights."""
    assert isinstance(weights, tf.Variable)
    # mask can be a variable (can be updated during training) or
    # .. can be a tensor (used as a constant mask)
    assert isinstance(mask, (tf.Variable, tf.Tensor))

    if weights in self.variable_dict:
      slot = self.variable_dict[weights]
      assert isinstance(slot, WeightSlot)
      return slot.masked_weights

    slot = WeightSlot(weights, mask=mask)
    self.variable_dict[weights] = slot
    self._dense_weights.append(slot)
    return slot.masked_weights

  def register_to_dense(self, weights, frac):
    """This method will be called only by linker.get_weights_to_prune."""
    assert frac > 0
    # In some case such as building in RNN, a same weights may be called twice
    # .. e.g. build_while_free -> _link
    if weights in self.variable_dict:
      slot = self.variable_dict[weights]
      assert isinstance(slot, WeightSlot)
      return slot.masked_weights

    slot = WeightSlot(weights, frac)
    self.variable_dict[weights] = slot
    self._dense_weights.append(slot)
    return slot.masked_weights

  def clear(self):
    # self._dense_weights = []
    # self._conv_filters = []
    # self.variable_dict = OrderedDict()
    pass

  @staticmethod
  def extractor(*args):
    if not any([tfr.hub.prune_on, tfr.hub.weights_mask_on,
                tfr.hub.export_weights]): return
    pruner = tfr.context.pruner
    if pruner is None:
      tfr.console.warning_with_pause(
        'Pruner.extractor has been called yet prune is not on')
      return
    for slot in pruner._dense_weights:
      assert isinstance(slot, WeightSlot)
      if tfr.hub.prune_on:
        tfr.context.variables_to_export[slot.weight_key] = slot.weights
        tfr.context.variables_to_export[slot.mask_key] = slot.mask
      elif tfr.hub.weights_mask_on:
        tfr.context.variables_to_export[slot.weight_key] = slot.masked_weights

  # endregion : Public Methods

  # region : Prune and save

  def prune_and_save(self):
    """This method should be called only in the end of trainer.train"""
    # pruning should start from best model
    assert tfr.hub.save_model

    tfr.console.show_status('Loading best model to prune ...')
    self._model.agent.load()

    self.prune()

    # This will force agent.ckpt_dir property to create path even if
    # .. current running model is not saved
    # tfr.hub.save_model = True

    self.save_next_model()

  def prune(self):
    # Get percentage
    assert self.th.prune_on
    p = tfr.hub.pruning_rate_fc
    assert 0 < p < 1.0
    # Fetch, prune and set
    self._fetch_masked_weights()
    tfr.console.show_status('Masked weights fetched. Pruning ...')
    self._set_weights_and_masks(p)
    # Update master fraction. Some layers, e.g. the output layer may have
    #  a different pruning rate
    self.dense_fraction = self.dense_fraction * (1 - p)
    # Show status
    tfr.console.show_status(
      'Dense fraction decreased to {:.2f}'.format(self.dense_fraction))

  def save_next_model(self):
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

  # endregion : Prune and save

  # region : Private Methods

  def _run_op(self, ops):
    from tframe.models.model import Model
    model = self._model
    assert isinstance(model, Model)
    return model.session.run(ops)

  def _fetch_masked_weights(self):
    masked_weights = [ws.masked_weights for ws in self._dense_weights]
    value_masked_weights = self._run_op(masked_weights)
    for vmw, ws in zip(value_masked_weights, self._dense_weights):
      assert isinstance(ws, WeightSlot)
      ws.value_masked_weights = vmw

  def _set_weights_and_masks(self, p):
    assert 0 < p < 1
    reset_weights_ops = [ws.reset_weights for ws in self._dense_weights]
    set_mask_ops = [
      ws.get_assign_mask_op(p) for ws in self._dense_weights]
    self._run_op([reset_weights_ops, set_mask_ops])
    # Show status
    tfr.console.show_status('Weights reset and masks updated.')

  def _fetch_masks(self):
    fetches = [ws.mask for ws in self._dense_weights]
    val_masks = self._run_op(fetches)
    # Distribute to weight_slots
    for slot, val_mask in zip(self._dense_weights, val_masks):
      assert isinstance(slot, WeightSlot)
      slot.value_mask = val_mask

  # endregion : Private Methods


class WeightSlot(object):
  """Weight Slot is not a tframe Slot"""

  def __init__(self, weights, frac=None, mask=None):
    # Sanity check
    assert isinstance(weights, tf.Variable)
    if frac is not None: assert np.isreal(frac) and 0 <= frac <= 1
    if mask is not None: assert isinstance(mask, (tf.Variable, tf.Tensor))

    self.weights = weights
    self.init_val = tf.Variable(
      tf.zeros_like(weights), trainable=False, name='init_val')

    if mask is not None: self.mask = mask
    else: self.mask = tf.Variable(
      tf.ones_like(weights), trainable=False, name='mask')

    self.masked_weights = tf.multiply(self.weights, self.mask)
    self.frac = frac

    # Define assign ops
    self.assign_init = tf.assign(self.init_val, self.weights)
    self.reset_weights = tf.assign(self.weights, self.init_val)

    # Define weights and mask placeholder
    self.value_masked_weights = None
    self.value_mask = None
    # weights fraction will be assigned during
    #  launching session => model.handle_structure_detail
    #  => net.structure_detail => pruner.get_variable_sizes
    self.weights_fraction = None

  @property
  def scope_abbr(self):
    """In tframe, RNN model, weight's name may be ../gdu/net_u/W"""
    scopes = self.weights.name.split('/')
    return '/'.join(scopes[-3:-1])

  @property
  def weight_key(self):
    """Used in Pruner.extractor"""
    scopes = self.weights.name.split('/')
    key = '/'.join(scopes[-3:])
    key = key.split(':')[0]
    return key

  @property
  def mask_key(self):
    return self.weight_key + '_mask'

  def get_assign_mask_op(self, p):
    assert 0 < p < 1
    p = p * self.frac
    w = self.value_masked_weights
    assert isinstance(w, np.ndarray) and np.isreal(self.weights_fraction)
    assert 0 < self.weights_fraction <= 100
    weight_fraction = np.ceil(self.weights_fraction * (1 - p))
    # Get weights magnitude
    w = np.abs(w)
    # Create mask
    mask = np.zeros_like(w, dtype=np.float32)
    mask[w > np.percentile(w, 100 - weight_fraction)] = 1.0
    # Return mask assign operator
    op = tf.assign(self.mask, mask)
    return op





from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf
import tframe as tfr


class Quantity(object):
  """Quantities mainly includes loss and metric defined for a model.
     Quantities are usually calculated as statistical results over
     a batch or the whole data set. Quantities are bound with TensorSlots

     Regardless of deep learning frames, a quantity (e.g. loss) is calculated
     as: input_batch -> model -> output_batch -> [loss(y) for y in output_batch]
         -> average function -> batch_loss (a scalar, or a quantity)

     In tframe, a batch validation, differ from one-shot validation which is
     done given some conditions are satisfied, works as:

                              input_batch_generator -> input_batch -> feed_dict
     -- GPU (or other tensorflow device) -----------------------------------│-
             denoted as                                                     ↓
     quantities <- [quantity(y) for y in output_batch] <- output_batch <- model
        ├-> last_only_checker -> tf_summ_method -> batch_quantity (1)
     - -↓- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      np_arrays -> active_length_checker -> last_only_checker -> output_list
                   np_summ_method <- after all results are gathered <-┘
                        └-> data_set_quantity (2)
     -- CPU -------------------------------------------------------------------

     (1) a node in tensorflow graph, will be fetched in every training step
     (2) will be returned by model.validate_model method
 """

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

    # This will be passed to metric in metric_manager.py -> initialize
    # This variable is first designed for instance segmentation metrics
    self.post_processor = kwargs.get('post_processor', None)

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
    # logits = tfr.context.logits_tensor_dict.get(output, None)
    logits = tfr.context.get_logits(output)
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
        raise AssertionError('!! tf_summ_method is provided but not used')
      if self._np_summ_method is not None:
        raise AssertionError('!! np_summ_method is provided but not used')
      if tfr.hub.use_batch_mask:
        raise AssertionError(
          '!! Batch mask can not be applied to quantity without summary')
      return q

    # Apply batch mask if provided
    if tfr.hub.use_batch_mask: q = q[tfr.hub.batch_mask]

    self._quantities = q
    # Extract result in last time step for RNN output
    # For SequenceSet containing non-equal-length sequences, q.shape[0] must
    #  be 1, i.e. batch_size must be 1
    if self._last_only:
      # q.shape must be [batch_size, steps, *dims]
      assert len(q.shape) > 1
      if tfr.hub.use_gather_indices:
        assert tfr.context.gather_indices is not None
        q = tf.gather_nd(q, tfr.context.gather_indices)
      else: q = q[:, -1]
    if self._tf_summ_method is None:
      if self.post_processor is None:
        raise TypeError('!! tf_summ_method should be provided')
      else:
        # For metric only
        assert callable(self.post_processor)
        return q
    self._quantity = self._tf_summ_method(q)
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

    # TODO: since this method works only for export tensors like dL/dS,
    #       last only logic is most likely to be suppressed. Otherwise,
    #       NaN issue will occur e.g. in export_dl_ds_stat process
    # Extract result in last time step for RNN output
    # if self._last_only:
    #   # In single step calculation, q does not have num_step dimension
    #   assert len(q.shape) > 0
    #   q = q[-1]

    if self._tf_summ_method is None:
      raise TypeError('!! summ_method should be provided')
    q = self._tf_summ_method(q)
    assert isinstance(q, tf.Tensor) and len(q.shape) == 0
    return q

  def _raise_not_linked_error(self):
    raise ValueError('!! This quantity has not been linked yet')

  def _check_link(self):
    if self._quantity is None: self._raise_not_linked_error()

  def apply_np_summ_method(self, quantities, show_detail=False):
    """Used only in model.validate_model method"""
    if not self._last_only:
      tfr.checker.check_type(quantities, np.ndarray)

    # Concatenate if necessary
    if isinstance(quantities, list):
      if show_detail: self._show_seq_metric_detail(quantities)
      # Extract data in last step if necessary
      # TODO: this has been done in __call__, check again
      # if self._last_only: quantities = [s[:, -1] for s in quantities]
      if isinstance(quantities[0], np.ndarray):
        quantities = np.concatenate(quantities, axis=0)

    if callable(self.np_summ_method): return self.np_summ_method(quantities)
    return quantities

  def _show_seq_metric_detail(self, val_list):
    console = tfr.console
    splits = tfr.hub.val_info_splits
    assert splits > 0
    size = int(np.ceil(len(val_list) / splits))
    console.show_info('Metric detail (BETA)')
    for i in range(splits):
      q = np.concatenate(val_list[i*size:(i+1)*size], axis=0)
      console.supplement('{:.3f}'.format(self.np_summ_method(q)))

  # region : Default Kernels

  @staticmethod
  def concate_dense_label_pred(label, pred):
    # Convert labels and outputs to 2-D dense tensors
    tensors = [label, pred]
    for i, tensor in enumerate(tensors):
      shape = tensor.shape.as_list()
      # Convert one-hot/distribution to dense if necessary
      if shape[-1] > 1:
        tensor = tf.argmax(tensor, -1, output_type=tf.int32)
        tensor = tf.expand_dims(tensor, -1)
      # Put tensor back to list
      tensors[i] = tensor
    # Concatenate for summary
    return tf.concat(tensors, axis=-1, name='label_pred')

  # endregion : Default Kernels





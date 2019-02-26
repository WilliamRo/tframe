from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import tensorflow as tf

from tframe import context
from tframe import hub

from tframe.models.model import Model
from tframe.nets import RNet
from tframe.layers import Input

from tframe.core.decorators import with_graph
from tframe.core import NestedTensorSlot

from tframe.utils.misc import transpose_tensor
from tframe.utils.misc import ravel_nested_stuff


class Recurrent(Model, RNet):
  """Recurrent neural network base class"""
  # TODO: where will this variable be used?
  model_name = 'RNN'

  def __init__(self, mark=None):
    Model.__init__(self, mark)
    RNet.__init__(self, 'RecurrentNet')
    self.superior = self
    self._default_net = self
    # Attributes
    self._state_slot = NestedTensorSlot(self, 'State')
    # mascot will be initiated as a placeholder with no shape specified
    # .. and will be put into initializer argument of tf.scan
    self._mascot = None
    self._while_loop_free_output = None

    # TODO: BETA
    self.last_scan_output = None
    self.grad_delta_slot = NestedTensorSlot(self, 'GradDelta')
    self._grad_buffer_slot = NestedTensorSlot(self, 'GradBuffer')

  # region : Properties

  @property
  def grad_buffer_slot(self):
    return self._grad_buffer_slot

  # endregion : Properties

  # region : Build

  def _build_while_free(self):
    assert isinstance(self.input_, Input)
    assert self.input_.rnn_single_step_input is not None
    input_placeholder = self.input_.rnn_single_step_input
    pre_outputs = (None, self.init_state)
    if hub.use_rtrl or hub.export_tensors_to_note: pre_outputs += (None,)

    # The mascot can also be used as the targets tensor placeholder
    self._mascot = tf.placeholder(dtype=hub.dtype, name='mascot')
    with tf.name_scope('OuterWhileLoop'):
      if self.loss_in_loop:
        input_placeholder = (input_placeholder, self._mascot)
      self._while_loop_free_output = self(pre_outputs, input_placeholder)

    # Plug targets tensor into targets slot if necessary
    if self.loss_in_loop:
      from tframe.models import Predictor
      assert isinstance(self, Predictor)
      y = self._while_loop_free_output[0]
      assert isinstance(y, tf.Tensor)
      shape = y.shape.as_list()
      self._plug_target_in([None] + shape)

    initializer = self._mascot, self.init_state
    def _get_nested_mascots(source):
      if isinstance(source, tf.Tensor):
        return self._mascot,
      else:
        assert isinstance(source, (list, tuple))
        mas_tuple = ()
        for src in source:
          mas_tuple += _get_nested_mascots(src)
        return mas_tuple,
    for output in self._while_loop_free_output[2:]:
      initializer += _get_nested_mascots(output)

    # for output in self._while_loop_free_output[2:]:
    #   if isinstance(output, tf.Tensor):
    #     initializer += self._mascot,
    #   else:
    #     assert isinstance(output, (list, tuple))
    #     initializer += ((self._mascot,) * len(output),)

    # Clear stuff
    context.clear_all_collections()
    return initializer

  @with_graph
  def _build(self, **kwargs):
    # self.init_state should be called for the first time inside this method
    #  so that it can be initialized within the appropriate graph

    # :: Define output
    # Make sure input has been defined
    if self.input_ is None: raise ValueError('!! input not found')
    assert isinstance(self.input_, Input)
    # Input placeholder has a shape of [batch_size, num_steps, *sample_shape]
    self.input_.set_group_shape((None, None))

    # Transpose input so as to fit the input of tf.scan
    input_placeholder = self.input_()

    # Build a shadow in order to foreknow the nested structure of `initializer`
    initializer = self._build_while_free()

    # Get elems to feed tf.scan
    elems = transpose_tensor(input_placeholder, [1, 0])
    if self.loss_in_loop:
      from tframe.models import Predictor
      assert isinstance(self, Predictor)
      targets_placeholder = self._targets.tensor
      elems = (elems, transpose_tensor(targets_placeholder, [1, 0]))

    # Send stuff into tf.scan and get results
    results = tf.scan(self, elems, initializer=initializer, name='Scan')
    scan_outputs, state_sequences = self._unwrap_outputs(results)

    # Activate state slot
    assert isinstance(self._state_slot, NestedTensorSlot)

    # Get last state and distribute to all recurrent-child
    last_state = Recurrent._extract_tensors(state_sequences, lambda t: t[-1])
    self._new_state_tensor = last_state
    self._distribute_last_tensors()

    # Plug last state to corresponding slot
    self._state_slot.plug(last_state)
    self._update_group.add(self._state_slot)

    # TODO: BETA
    if hub.use_rtrl: self._update_group.add(self.grad_buffer_slot)
    if hub.test_grad: self._update_group.add(self.grad_delta_slot)

    # Transpose scan outputs to get final outputs
    outputs = transpose_tensor(scan_outputs, [1, 0])

    # Output has a shape of [batch_size, num_steps, *output_shape]
    self.outputs.plug(outputs)

  @staticmethod
  def _extract_tensors(tensors, extract):
    """This method is used specifically for the tf.scan output"""
    if isinstance(tensors, (list, tuple)):
      last_state = []
      for obj in tensors:
        last_state.append(Recurrent._extract_tensors(obj, extract))
      return last_state
    else:
      assert isinstance(tensors, tf.Tensor)
      return extract(tensors)

  def _unwrap_outputs(self, results):
    results = list(results)
    # 1. Outputs
    y = results.pop(0)
    # 2. States
    state = results.pop(0)
    # 3. Logits
    if self.logits_tensor is not None:
      self._logits_tensor = transpose_tensor(results.pop(0), [1, 0])
    # 4. Tensors to export
    if hub.export_tensors_to_note:
      self._set_tensors_to_export(
        [transpose_tensor(t, [1, 0]) for t in results.pop(0)])
    # 5. Extra losses
    losses = []
    if context.loss_tensor_list:
      losses.append(tf.reduce_sum(results.pop(0)))
    if callable(context.customed_outer_loss_f_net):
      customized_losses = self._get_customized_loss(outer=True)
      if hub.show_extra_loss_info:
        print(':: {} outer losses added.'.format(len(customized_losses)))
      losses += customized_losses
    if len(losses) == 1:
      self._extra_loss = losses[0]
    elif losses:
      self._extra_loss = tf.add_n(losses, name='extra_loss')

    # 6. TODO: BETA
    if hub.use_rtrl:
      self._grad_tensors = self._extract_tensors(
        results.pop(0), lambda t: t[-1])
      self.last_scan_output = y

    # 7. Loss related tensors
    if self.loss_in_loop and hub.export_dl_dx:
      # Define extraction function
      f = lambda t: t[:, 0]
      # each dL_t/dS_{t-1} \in R^{num_steps, batch_size, state_size}
      dl_dsp = self._extract_tensors(results.pop(0), f)
      # each dS_t/dS_{t-1} \in R^{num_steps, batch_size, state_size, state_size}
      ds_dsp = self._extract_tensors(results.pop(0), f)
      # Register tensors
      # TODO: if can be calculated here, {dL(t)/dS(\tau)}_{t, \tau} should be
      #       added to context.tensors_to_export
      export_dict = context.tensors_to_export
      od = self._get_dL_dS_dict(dl_dsp, ds_dsp)
      for k, v in od.items():
        export_dict[k] = v

    # Return
    assert len(results) == 0
    return y, state

  def _get_dL_dS_dict(self, dlds_nested, dsds_nested):
    dlds_flat, _ = ravel_nested_stuff(dlds_nested, with_indices=True)
    dsds_flat, indices = ravel_nested_stuff(dsds_nested, with_indices=True)
    od = OrderedDict()
    # Keys are '(dL/dSi)j'
    for dlds, dsds, index in zip(dlds_flat, dsds_flat, indices):
      assert isinstance(index, list)
      assert isinstance(dlds, tf.Tensor) and isinstance(dsds, tf.Tensor)
      assert len(dlds.shape) == 2 and len(dsds.shape) == 3
      # Generate key for dL/dSi
      if len(dlds_flat) == 1: grad_name = 'S'
      else: grad_name = 'S{}'.format('-'.join([str(i + 1) for i in index]))
      grad_name = '(dL/d{})'.format(grad_name)

      # Say T = num_steps, (dL/dSi)j is a T by T lower triangular matrix
      triangle = self._form_triangle(dlds, dsds)
      od[grad_name + '*'] = tf.reduce_sum(
        tf.abs(triangle), axis=0, keepdims=True)
      assert isinstance(triangle, tf.Tensor)
      for i, t in enumerate(tf.split(triangle, triangle.shape.as_list()[0])):
        od['{}{}'.format(grad_name, i + 1)] = t

    return od

  def _form_triangle(self, dlds, dsds):
    assert isinstance(dlds, tf.Tensor) and isinstance(dsds, tf.Tensor)
    assert len(dlds.shape) == 2 and len(dsds.shape) == 3
    T = tf.shape(dlds)[0]

    def body_outer(i, rows):
      """Calculates the i-th row of the triangle (not strictly).
         The shape of the row should be [T-1, state_size]
      """
      assert isinstance(rows, tf.TensorArray)
      # Initialize grad
      grad_init = dlds[i]
      assert isinstance(grad_init, tf.Tensor) and len(grad_init.shape) == 1
      grad_init = tf.reshape(grad_init, [-1, 1])

      def body_inner(j, row, grad):
        assert isinstance(row, tf.TensorArray)
        grad_to_write = tf.transpose(grad, [1, 0])

        # Write grad to row
        row = row.write(j - 1, tf.cond(
          tf.less_equal(j, i), lambda: grad_to_write,
          lambda: tf.zeros_like(grad_to_write)))
        # Update grad and return
        grad = tf.matmul(dsds[j-1], grad)
        return j - 1, row, grad

      # Use tf.while_loop to generate row
      empty_row = tf.TensorArray(tf.float32, size=T-1, dynamic_size=False)
      _, row, _ = tf.while_loop(
        lambda j, *_: tf.greater(j, 0), body_inner,
        (T - 1, empty_row, grad_init), back_prop=False)
      # each entry in row has a shape [1, state_size]
      assert isinstance(row, tf.TensorArray)
      return i + 1, rows.write(i - 1, row.concat(name='row'))

    ta_outer_init = tf.TensorArray(tf.float32, size=T-1)
    _, rows = tf.while_loop(
      lambda i, *_: tf.less(i, T), body_outer, (1, ta_outer_init),
      back_prop=False)

    assert isinstance(rows, tf.TensorArray)
    # each row has a shape [T-1, state_size]
    triangle = rows.stack()
    return tf.transpose(triangle, [2, 0, 1], name='triangle')

  # endregion: Build

















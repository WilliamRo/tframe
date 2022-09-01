from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf

from tframe.models.model import Model
from tframe.models.feedforward import Feedforward
from tframe.models.recurrent import Recurrent

from tframe import console
from tframe import checker
from tframe import context
from tframe import losses
from tframe import pedia
from tframe import metrics

from tframe import hub
from tframe import InputTypes
from tframe.core import with_graph
from tframe.core import TensorSlot
from tframe.core.quantity import Quantity


class Predictor(Feedforward, Recurrent):
  """A feedforward or a recurrent predictor"""
  model_name = 'Predictor'

  def __init__(self, mark=None, net_type=Feedforward):
    """
    Construct a Predictor
    :param mark: model mark
    :param net_type: \in {Feedforward, Recurrent}
    """
    if not net_type in (Feedforward, Recurrent):
      raise TypeError('!! Unknown net type')
    self.master = net_type
    # Attributes
    self._targets = TensorSlot(self, 'targets')
    self._val_targets = TensorSlot(self, 'val_targets')
    # Call parent's constructor
    net_type.__init__(self, mark)

  # region : Properties

  @property
  def affix(self):
    if self.master is Feedforward: return 'forward'
    assert self.master is Recurrent
    return 'recurrent'

  @property
  def description(self):
    return '{}: {}'.format(self.master.__name__, self.structure_string())

  @property
  def input_type(self):
    if self.master is Feedforward: return InputTypes.BATCH
    else: return InputTypes.RNN_BATCH

  # endregion : Properties

  # region : Build

  @with_graph
  def build_as_regressor(
      self, optimizer=None, loss='euclid', metric='rms_ratio',
      metric_name='Err %'):
    self.build(
      optimizer=optimizer, loss=loss, metric=metric, metric_name=metric_name)

  @with_graph
  def build(self, optimizer=None, loss='euclid', metric=None,
            batch_metric=None, eval_metric=None, **kwargs):
    context.metric_name = 'unknown' # TODO: to be deprecated
    Model.build(self, optimizer=optimizer, loss=loss, metric=metric,
                batch_metric=batch_metric, eval_metric=eval_metric, **kwargs)

  def _build(self, optimizer=None, loss='euclid', metric=None, **kwargs):
    # For some RNN predictors, their last step is counted as the only output
    #   e.g. RNNs for sequence classification tasks
    last_only = False
    if 'last_only' in kwargs.keys():
      last_only = kwargs.pop('last_only')
      if hub.use_gather_indices and last_only:
        # Initiate gather_indices placeholder
        assert context.gather_indices is None
        context.gather_indices = tf.placeholder(
          tf.int32, [None, 2], 'gather_indices')
        tf.add_to_collection(pedia.default_feed_dict, context.gather_indices)

    # Get loss quantity before building
    self.loss_quantity = losses.get(loss, last_only)
    # This is for calculating loss inside a while-loop
    # 2020-6-10 | William |
    #   this line is solely used for visualizing gradients
    #   thus does not affect error injection in RNNs
    context.loss_function = self.loss_quantity.function

    # Call parent's build method to link network
    # Usually output tensor has been plugged into Model._outputs slot
    self.master._build(self)
    assert self.outputs.activated

    # Initiate targets and add it to collection
    self._plug_target_in(self.outputs.shape_list)

    # Initialize shadow vars if necessary
    self._init_shadows()

    # Initialize variables for continual learning regularizer calculation
    if hub.cl_reg_on:
      context.puller.spring.init_after_linking_before_calc_loss()

    # Define loss. Some tensorflow apis only support calculating logits
    with tf.name_scope('Loss'):
      # (1) main loss
      loss_tensor = self.loss_quantity(
        self._targets.tensor, self.outputs.tensor)

      # TODO: with or without regularization loss?
      if hub.summary:
        tf.add_to_collection(pedia.train_step_summaries,
                             tf.summary.scalar('loss_sum', loss_tensor))
      # (2) Try to add extra loss which is calculated by the corresponding net.
      #     regularization loss is included.
      if self.extra_loss is not None:
        loss_tensor = tf.add(loss_tensor, self.extra_loss)
      # (3) Add additional injected loss if necessary
      injection_loss = self._gen_injection_loss()
      if injection_loss is not None:
        loss_tensor = tf.add(loss_tensor, injection_loss)

      # Plug in
      self.loss.plug(loss_tensor, quantity_def=self.loss_quantity)

    # <monitor_grad_step_02: register loss and plug grad_ops in>
    if hub.monitor_weight_grads:
      context.monitor.register_loss(loss_tensor)
      self.grads_slot.plug(context.monitor.grad_ops_list)
      self._update_group.add(self.grads_slot)

    # Monitor general tensors (currently only activation is included)
    if hub.export_activations and context.monitor.tensor_fetches:
      self.general_tensor_slot.plug(context.monitor.tensor_fetches)
      self._update_group.add(self.general_tensor_slot)

    # Initialize metric
    if metric is not None:
      checker.check_type_v2(metric, (str, Quantity))
      # Create placeholder for val_targets if necessary
      # Common targets will be plugged into val_target slot by default
      self._plug_val_target_in(kwargs.get('val_targets', None))

      with tf.name_scope('Metric'):
        self._metrics_manager.initialize(
          metric, last_only, self._val_targets.tensor,
          self.val_outputs.tensor, **kwargs)

    # Merge summaries
    self._merge_summaries()

    # Define train step
    self._define_train_step(optimizer)

  def _plug_target_in(self, shape):
    dtype = hub.dtype
    if hub.target_dim != 0: shape[-1] = hub.target_dim
    elif hub.target_shape not in ('-', 'null'): shape = hub.target_shape
    # If target is sparse label
    if hub.target_dtype is not None: dtype = hub.target_dtype
    # if hub.target_dim == 1: dtype = tf.int32  # TODO: X

    # Handle recurrent situation
    if self._targets.tensor is not None:
      # targets placeholder has been plugged in Recurrent._build_while_free
      #   method
      assert self.master == Recurrent
      return
    target_tensor = tf.placeholder(dtype, shape, name='targets')
    self._targets.plug(target_tensor, collection=pedia.default_feed_dict)

  def _plug_val_target_in(self, val_targets):
    """TODO: logic of val_targets has been borrowed for supporting
       non-train-input logic. Need to be refactored
    """
    # TODO: to be refactored
    if val_targets is None and hub.non_train_input_shape is None:
      self._val_targets = self._targets
    else:
      # assert isinstance(val_targets, str)
      val_target_tensor = tf.placeholder(
        hub.dtype, self.val_outputs.shape_list, name='val_targets')
        # hub.dtype, self.outputs.shape_list, name = 'val_targets')  # TODO
      self._val_targets.plug(
        val_target_tensor, collection=pedia.default_feed_dict)

  # endregion : Build

  # region : Train

  def update_model(self, data_batch, **kwargs):
    if self.master is Feedforward:
      return Feedforward.update_model(self, data_batch, **kwargs)
    # Update recurrent model
    feed_dict = self._get_default_feed_dict(data_batch, is_training=True)
    results = self._update_group.run(feed_dict)
    self.set_buffers(results.pop(self._state_slot), is_training=True)

    # TODO: BETA
    assert not hub.use_rtrl
    if hub.use_rtrl:
      self._gradient_buffer_array = results.pop(self._grad_buffer_slot)
    if hub.test_grad:
      delta = results.pop(self.grad_delta_slot)
      _ = None
    return results

  # endregion : Train

  # region : Public Methods

  @with_graph
  def predict(self, data, batch_size=None, extractor=None, **kwargs):
    return self.evaluate(
      self.val_outputs.tensor, data, batch_size, extractor, **kwargs)

  @with_graph
  def evaluate_model(self, data, batch_size=None, dynamic=False, **kwargs):
    """The word `evaluate` in this method name is different from that in
       `self.evaluate` method. Here only eval_metric will be evaluated and
       the result will be printed on terminal."""
    # Check metric
    if not self.eval_metric.activated:
      raise AssertionError('!! Metric not defined')
    # Do dynamic evaluation if necessary
    if dynamic:
      from tframe.trainers.eval_tools.dynamic_eval import DynamicEvaluator as de
      de.dynamic_evaluate(
        self, data, kwargs.get('val_set', None), kwargs.get('delay', None))
      return
    # If hub.val_progress_bar is True, this message will be showed in
    #   model.evaluate method
    if not hub.val_progress_bar:
      console.show_status('Evaluating on {} ...'.format(data.name))
    # use val_progress_bar option here temporarily
    result = self.validate_model(
      data, batch_size, allow_sum=False,
      verbose=hub.val_progress_bar)[self.eval_metric]
    console.supplement('{} = {}'.format(
      self.eval_metric.symbol, hub.decimal_str(result, hub.val_decimals)))

    return result

  # endregion : Public Methods

  # region : Private Methods

  def _evaluate_batch(self, fetch_list, data_batch, **kwargs):
    return self.master._evaluate_batch(self, fetch_list, data_batch, **kwargs)

  def _get_default_feed_dict(self, batch, is_training):
    return self.master._get_default_feed_dict(self, batch, is_training)

  # endregion : Private Methods

  # region: Visualization

  def visualize_tensors(
      self, data_set, tensor_dict: dict = None, max_tensors=None,
      max_channels=None, visualize_kernels=False):
    """Visualize tensors related to 2-D convolutional networks"""
    from tframe.data.dataset import DataSet
    from lambo.gui.vinci.vinci import DaVinci
    from tframe.utils.display.img_utils import tile_kernels
    from collections import OrderedDict

    # Sanity check
    assert isinstance(data_set, DataSet)
    assert tensor_dict is None

    # Find tensors to visualize. Shape of each tensor should be
    #  [bs, H, W, C] for layer output or [bs, K, K, c_in, c_out] for kernels
    if tensor_dict is None:
      tensor_dict = OrderedDict()
      tensor_list = (context.get_collection_by_key(pedia.hyper_kernels)
                     if visualize_kernels else [net.output_tensor
                                                for net in self.children])
      tensor_dict['Input'] = (self.input_tensor if self._shadow_input is None
                              else self._shadow_input.output_tensor)
      for tensor in tensor_list:
        tensor_dict[tensor.name.split('/')[1]] = tensor
        if max_tensors is not None and len(tensor_dict) == max_tensors: break

    assert isinstance(tensor_dict, dict) and len(tensor_dict) > 0

    # Get tensor
    values = self.evaluate(
      list(tensor_dict.values()), data_set, batch_size=1, verbose=True)

    # Constructor DaVinci
    da = DaVinci('Tensor Visualizer', height=7, width=7)
    da._k_space_log = True
    # Fill in objects
    batch_size = len(values[0])
    for i in range(batch_size):
      vs = [v[i] for v in values]
      # If tensor is kernel, tile them
      if len(vs[0].shape) == 4: vs = [tile_kernels(v) for v in vs]
      # Append vs to objects
      da.objects.append(vs)

    # Set
    def imshow(net_id, slice_id, total):
      def _imshow(x):
        net_name = list(tensor_dict.keys())[net_id]
        x = x[net_id][:, :, slice_id]
        title = '{}[{}/{}] Shape = {}x{}'.format(
          net_name, slice_id + 1, total, x.shape[0], x.shape[1])
        da.imshow_pro(x, title=title)
      return _imshow

    # Add plotter
    anchors = []
    for i, v in enumerate(da.objects[0]):
      total = v.shape[-1]
      anchors.append(len(da.layer_plotters))
      for j in range(total):
        if max_channels is not None and j == max_channels: break
        da.add_plotter(imshow(i, j, total))

    # Add fast forward/backward function
    def fast_for_back_ward(d):
      assert d in (-1, 1)
      a = anchors[::d]
      for li in a:
        if d * (li - da.layer_cursor) > 0:
          da.layer_cursor = li
          return
      da.layer_cursor = a[0]

    da.state_machine.register_key_event('H', lambda: fast_for_back_ward(-1))
    da.state_machine.register_key_event('L', lambda: fast_for_back_ward(1))

    # Set method for exporting videos
    def dump(fps: float = 1.0, fmt: str = 'mp4', folder_name: str = None):
      import os

      # Find path to dump
      if folder_name is None: folder_name = 'variables-{}'.format(self.rounds)
      directory = os.path.join(self.agent.ckpt_dir, folder_name)
      if not os.path.exists(directory): os.mkdir(directory)
      console.show_status('Dumping variables to `{}` ...'.format(directory))

      # Dump
      index = 1
      for i, v in enumerate(da.objects[0]):
        net_name = list(tensor_dict.keys())[i]
        n_pages = v.shape[-1]
        if max_channels is not None: n_pages = min(n_pages, max_channels)

        # TODO: allow exporting input
        # if 'input' in net_name.lower():
        #   index += n_pages
        #   continue

        console.show_status('Generating animation for {} ({} frames)'.format(
          net_name, n_pages))

        cursor_range = '{}:{}'.format(index, index + n_pages - 1)
        fn = 'o{}-{}-{}-{}'.format(
          da.object_cursor + 1, 'F' if da._k_space else 'I', i + 1, net_name)
        p = os.path.join(directory, fn)
        da.export('l', fps, cursor_range, fmt, path=p, n_tail=5)
        index += n_pages

      console.show_status('Variables dumped to `{}` ...'.format(directory))

    da.dump = dump

    # Show
    da.show()

  # endregion: Visualization








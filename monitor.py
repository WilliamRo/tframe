from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import pedia
from tframe import hub
from tframe.utils.misc import get_name_by_levels
from tframe.core import OperationSlot, SummarySlot, TensorSlot, Group


class Monitor(object):
  """A MONITOR monitors model status during training. Each tframe environment
     (process) has only one monitor. A monitor shows its results via
     tensorboard. Currently monitor only supports weights in fc layers"""
  def __init__(self):
    # Private attributes
    self._update_ops = []
    self._grad_lounge = []
    self._weight_lounge = []
    self._postact_lounge = []
    self._round_end_summaries = []
    self._round_end_group = None

    # Public attributes
    self.decay = 0.99

  # region : Properties

  @property
  def activated(self):
    return hub.monitor_grad or hub.monitor_weight or hub.monitor_postact

  # endregion : Properties

  # region : Private Methods

  def _create_shadow(self, op, name=None):
    assert isinstance(op, (tf.Tensor, tf.Variable))
    shadow = tf.Variable(initial_value=tf.zeros_like(op), name=name)
    tf.add_to_collection(pedia.do_not_save, shadow)
    update = tf.assign(shadow, self.decay * shadow + (1 - self.decay) * op)
    self._update_ops.append(update)
    return shadow

  @staticmethod
  def _make_image_summary(tensor, name=None):
    assert isinstance(tensor, (tf.Tensor, tf.Variable))
    if name is None: name = tensor.name
    shape = tensor.shape.as_list()
    # Currently tensors of 2-D are supported only
    assert len(shape) == 2
    if shape[1] == 1:
      # Make vector square
      tensor = tf.concat([tensor] * shape[0], axis=1)
      shape = tensor.shape.as_list()
    image = tf.reshape(tf.abs(tensor), [1] + shape + [1])
    # Initiate an image summary for image tensor
    image_summary = tf.summary.image(name, image, max_outputs=1)
    return image_summary

  @staticmethod
  def _get_default_name(op, group):
    assert isinstance(op, (tf.Tensor, tf.Variable))
    assert isinstance(group, str)
    return '{}/{}'.format(group, get_name_by_levels(op.name, (0, 1)))

  def _receive_weight_grad(self, loss):
    assert isinstance(loss, tf.Tensor)
    if len(self._grad_lounge) == 0: return
    for grad, weight in zip(tf.gradients(loss, self._grad_lounge),
                            self._grad_lounge):
      shadow = self._create_shadow(tf.abs(grad))
      self._round_end_summaries.append(self._make_image_summary(
        shadow, self._get_default_name(weight, 'Weight_Grads')))

  def _receive_post_activation(self):
    for tensor in self._postact_lounge:
      assert isinstance(tensor, tf.Tensor)
      mean = tf.reshape(tf.reduce_mean(tf.abs(tensor), axis=0), shape=[-1, 1])
      shadow = self._create_shadow(mean)
      self._round_end_summaries.append(self._make_image_summary(
        shadow, self._get_default_name(tensor, 'Post_Activation')))

  # endregion : Private Methods

  # region : Public Methods

  def add_preact_summary(self, tensor):
    assert isinstance(tensor, tf.Tensor)
    with tf.name_scope('preact_monitor'):
      pre_act = tf.summary.histogram('pre-activation', tensor)
      # Once added to collection, summaries will be merged in the building stage
      # .. of a model if model._merge_summaries has been called
      tf.add_to_collection(pedia.train_step_summaries, pre_act)

  def add_postact_summary(self, tensor):
    self._postact_lounge.append(tensor)

  def add_weight(self, weight):
    assert isinstance(weight, tf.Variable)
    # Monitor weight itself
    if hub.monitor_weight:
      self._weight_lounge.append(weight)
    # Monitor gradient
    if hub.monitor_grad:
      self._grad_lounge.append(weight)

  def init_monitor(self, model):
    from tframe.models import Model
    assert isinstance(model, Model)
    # (2) Post-activation reception
    self._receive_post_activation()
    # (3) Weight reception
    for weight in self._weight_lounge:
      self._round_end_summaries.append(self._make_image_summary(
        weight, self._get_default_name(weight, 'Weights')))
    # (4) Add gradients of loss with respect to each weight variable
    self._receive_weight_grad(model.loss.tensor)
    # (*) Wrap and register update_ops
    for op in self._update_ops:
      slot = OperationSlot(model)
      slot.plug(op)
      model._update_group.add(slot)
    # Organize round_end_group
    if len(self._round_end_summaries) > 0:
      slot = SummarySlot(model)
      slot.plug(tf.summary.merge(self._round_end_summaries))
      self._round_end_group = Group(model, slot)

  def export(self):
    if self._round_end_group is None: return
    assert isinstance(self._round_end_group, Group)
    self._round_end_group.run()

  # endregion : Public Methods

  """
  ---------------------------------------------------------------------------
  Researcher demands:
  ---------------------------------------------------------------------------
  (1) Monitoring the distributions of tensors before they flow into activation 
      layers (as histogram summaries)
  (2) Monitoring the running averages of the absolute value of tensors right 
      after the activation layers (as image summaries)
  (3) Monitoring the weights as snapshots after each training round
  (4) Monitoring the running averages of the gradients of training loss with 
      respect to each weight variable in the given model
  
  ---------------------------------------------------------------------------
  Life cycle of a monitor
  ---------------------------------------------------------------------------
  * Instantiated in a tframe process as the one and only instance of Monitor 
    class 
  * The corresponding flags are turned on. e.g. 'th.monitor = True' (the 
    global flag is set to True)
  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  * During model building stage:
    * (1) pre-activation info is wrapped as a histogram summary and added to 
          'train_step_summaries' collection
    * (2) post-activation info is tracked by its absolute mean shadow and for
          each shadow, an image summary is created and added to the 
          corresponding waiting list
    * (3) an image summary is created based on weight variables themselves 
          and is added to the corresponding waiting list
    * (4) the weight variables are directly added to a corresponding waiting 
          list
  * During the core building stage of a model:
    * (1) the histogram summaries will be wrapped into a summary slot and 
          registered into the default update_group of the model
  * After the model has been built preliminarily:
    * (4) the gradients are calculated and the corresponding image summaries
          are created
    * (*) the running average update operations are registered to the default
          update group of the model
  
  """


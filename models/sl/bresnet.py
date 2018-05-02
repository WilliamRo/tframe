from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe import console
from tframe import Predictor
from tframe import losses
from tframe import metrics
from tframe import pedia

from tframe.core import with_graph
from tframe.core import OperationSlot, TensorSlot
from tframe.core import Group
from tframe.trainers import Metric
from tframe.models import Feedforward
from tframe.nets.net import Net


class BResNet(Predictor):
  """Branch-Residual Net...? Fine, it's just a temporary name."""
  def __init__(self, mark=None):
    # Call parent's initializer
    Predictor.__init__(self, mark)
    # Private attributes
    # .. Options
    self.strict_residual = True
    # .. tframe objects
    self._master = 0
    self._boutputs = []
    self._losses = []
    self._train_steps = []
    self._metrics = []

  # region : Properties

  @property
  def num_branches(self):
    return len(self._boutputs)

  @property
  def record(self):
    if len(self._metrics) == 0: return None
    else: return max([metric.record for metric in self._metrics])

  # endregion : Properties

  # region : Build

  @with_graph
  def _build(self, loss='cross_entropy', optimizer=None,
             metric=None, metric_is_like_loss=True, metric_name='Metric'):
    Feedforward._build(self)
    # Check shapes of branch outputs
    output_shape = self._check_branch_outputs()
    # Initiate targets placeholder
    self._plug_target_in(output_shape)

    # Define output tensors
    for i, output in enumerate(self.branch_outputs):
      if i == 0 or not self.strict_residual:
        output_tensor = output
      else: output_tensor = output + self._boutputs[i - 1].tensor
      slot = TensorSlot(self, name='output_{}'.format(i + 1))
      slot.plug(output_tensor)
      self._boutputs.append(slot)

    # Define loss tensors
    loss_function = losses.get(loss)
    with tf.name_scope('Loss'):
      for i, output in enumerate(self._boutputs):
        assert isinstance(output, TensorSlot)
        slot = TensorSlot(self, name='loss_{}'.format(i + 1))
        slot.plug(loss_function(self._targets.tensor, output.tensor))
        self._losses.append(slot)

    # Define metric tensors
    metric_function = metrics.get(metric)
    if metric_function is not None:
      with tf.name_scope('Metric'):
        for i, output in enumerate(self._boutputs):
          assert isinstance(output, TensorSlot)
          slot = Metric(self, name='metric_{}'.format(i + 1))
          slot.plug(metric_function(self._targets.tensor, output.tensor),
                    as_loss=metric_is_like_loss,
                    symbol='{}{}'.format(metric_name, i + 1))
          self._metrics.append(slot)

    # Define train step
    self._define_train_step(optimizer)

    # Define groups
    self._update_group = Group(self, *self._losses, *self._train_steps)
    self._validate_group = Group(self, *self._metrics)

  def _check_branch_outputs(self):
    # Make sure at least one branch exists
    num_branches = len(self.branch_outputs)
    if num_branches < 1: raise AssertionError('!! No branch outputs found')
    # Get the 1st branch output shape
    branch_0 = self.branch_outputs[0]
    assert isinstance(branch_0, tf.Tensor)
    shape = branch_0.shape.as_list()
    # Check the remaining branch output
    for branch_output in self.branch_outputs[1:]:
      assert isinstance(branch_output, tf.Tensor)
      if branch_output.shape.as_list() != shape:
        raise ValueError('!! Each branch should have the same output shape')
    return shape

  def _define_train_step(self, optimizer=None, var_list=None):
    assert len(self._losses) > 0
    with tf.name_scope('Optimizer'):
      if optimizer is None: optimizer = tf.train.AdamOptimizer(1e-4)
      self._optimizer = optimizer
      loss_index = 0
      var_list = []
      for i, net in enumerate(self.children):
        assert isinstance(net, Net)
        var_list += net.var_list
        if net.is_branch or self._inter_type == pedia.fork:
          slot = OperationSlot(
            self, name='train_step_{}'.format(loss_index + 1))
          slot.plug(optimizer.minimize(
            loss=self._losses[loss_index].tensor, var_list=var_list))
          self._train_steps.append(slot)
          loss_index += 1
          var_list = []
    assert len(self._losses) == len(self._train_steps)

  # endregion : Build

  # region : Train

  def pretrain(self, **kwargs):
    # Try to use train scheme
    if self._scheme is not None:
      super().pretrain()
      return
    # Use default scheme
    self._master = kwargs.get('start_at', 0)
    for i, loss in enumerate(self._losses):
      assert isinstance(loss, TensorSlot)
      loss.sleep = i < self._master
    for i, train_step in enumerate(self._train_steps):
      assert isinstance(train_step, OperationSlot)
      train_step.sleep = i < self._master
    for i, metric in enumerate(self._metrics):
      assert isinstance(metric, Metric)
      metric.sleep = i < self._master
    self._metric = self._metrics[self._master]

  def bust(self, rnd):
    # Try to use scheme
    if self._scheme is not None:
      return super().bust(rnd)
    # Use default scheme
    if self._master + 1 == self.num_branches: return True
    # Let the busted branch sleep
    self._losses[self._master].sleep = True
    self._train_steps[self._master].sleep = True
    self._metrics[self._master].sleep = True
    self._master += 1
    # Set master metric
    master_metric = self._metrics[self._master]
    assert isinstance(master_metric, Metric)
    self._metric = master_metric
    # TODO
    self._metric._record_round = rnd

    return False

  def take_down_metric(self):
    for i, metric in enumerate(self._metrics):
      assert isinstance(metric, Metric) and metric.activated
      notes = 'Branch {}: Record: {:.3f}, Mean Record: {:.3f}'.format(
        i + 1, metric.record, metric.mean_record)
      self.agent.take_notes(notes, date_time=False)

  def end_round(self, rnd):
    for i, metric in enumerate(self._metrics):
      assert isinstance(metric, Metric) and metric.activated
      if metric.sleep: continue
      console.write_line('Branch {}  {}'.format(i + 1, '- ' * 35))
      metric.end_round(rnd)
    console.write_line('- ' * 40)

  # endregion : Train

  # region : Public Methods

  def predict(self, data, branch_index=0, additional_fetches=None):
    self._outputs.plug(self._boutputs[branch_index].tensor)
    return Predictor.predict(self, data, additional_fetches)

  # endregion : Public Methods









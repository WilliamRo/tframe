from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np

from tframe import tf
from tframe import console
from tframe import checker
from tframe import pedia
from tframe import hub as th
from tframe.models.model import Model

from tframe.data.dataset import DataSet
from tframe.data.sequences.seq_set import SequenceSet
from tframe.core.quantity import Quantity


class KrauseEvaluator(object):
  """Optimizer for dynamic evaluation using RMS with an RMS global prior
     References:
       [1] Ben Krause, etc. Dynamic Evaluation of Neural Sequence Models. 2018.
       [2] https://github.com/benkrause/dynamic-evaluation/blob/master/dynamiceval.py
  """

  # hyper-parameter values to be searched (see [2])
  # lrlist = [0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.0001]
  lrlist = [0.00001, 0.00002, 0.00003]
  # lamblist = [0.001, 0.002, 0.003, 0.005]
  # On cPTB, lambda should be large
  lamblist = [0.05, 0.1, 0.15, 0.2]

  PROMPT = '[Krause]'
  show_status = lambda _, s: console.show_status(s, KrauseEvaluator.PROMPT)

  def __init__(self, model, metric_quantity=None, epsilon=0.00002):
    assert isinstance(model, Model)
    self._model = model
    self._metric_quantity = metric_quantity
    #
    self._epsilon = checker.check_type(epsilon, float)
    self._eta = 0.00005
    self._eta_placeholder = tf.placeholder(tf.float32, name='de_eta')
    self._tf_eta = tf.Variable(
      initial_value=0.0, trainable=False, collections=[pedia.do_not_save],
      name='de_lr', dtype=tf.float32)
    self._set_eta_op = tf.assign(self._tf_eta, self._eta_placeholder)

    self._lambda = 0.002
    self._lambda_placeholder = tf.placeholder(tf.float32, name='de_lambda')
    self._tf_lambda = tf.Variable(
      initial_value=0.0, trainable=False, collections=[pedia.do_not_save],
      name='decay_lambda', dtype=tf.float32)
    self._set_lambda_op = tf.assign(self._tf_lambda, self._lambda_placeholder)

    self._var_list = []
    self._theta_0 = {}
    self._reset_theta_ops = None
    self._save_theta0_ops = None

    self._sqrt_MS_g = {}
    self._decay_rate_buffer = {}

    self._avg_sqrt_MS_g = None
    self._RMS_norm_buffer = {}

    self._grads = {}

    # Create slots
    self._create_slots()
    self._model.launch_model(overwrite=False)
    self._model.session.run(self._save_theta0_ops)
    console.show_status('Global parameters saved.', '[Krause]')

  # region : Properties

  @property
  def _RMS_norm(self):
    """This property is corresponding to decrate in [2]"""
    if len(self._RMS_norm_buffer) > 0:
      assert len(self._RMS_norm_buffer) == len(self._var_list)
      assert self._avg_sqrt_MS_g is not None
      return self._RMS_norm_buffer
    # Initialize RMS_norm_buffer
    self._init_RMS_norm()
    return self._RMS_norm

  @property
  def _decay_rate(self):
    """Decay rate should not exceed 1 for any parameters"""
    if len(self._decay_rate_buffer) > 0: return self._decay_rate_buffer
    self._decay_rate_buffer = {
      v: tf.minimum(1.0, self._tf_lambda * RMS_norm, 'decay_rate')
      for v, RMS_norm in self._RMS_norm.items()}
    return self._decay_rate

  # endregion : Properties

  # region : Private Methods

  def _init_RMS_norm(self):
    assert len(self._sqrt_MS_g) > 0
    self._avg_sqrt_MS_g = tf.reduce_mean([
      tf.reduce_mean(sqrt_MS_g) for sqrt_MS_g in self._sqrt_MS_g.values()])
    self._RMS_norm_buffer = {v: sqrt_MS_g / self._avg_sqrt_MS_g
                             for v, sqrt_MS_g in self._sqrt_MS_g.items()}

  def _create_slots(self, var_list=None):
    if var_list is None: var_list = tf.trainable_variables()
    self._var_list = var_list
    with tf.name_scope('de_theta0'):
      self._theta_0 = {
        theta: tf.Variable(theta, trainable=False) for theta in var_list}
    with tf.name_scope('de_sqrt_MS_g'):
      self._sqrt_MS_g = {
        theta: tf.Variable(tf.zeros_like(theta), trainable=False)
        for theta in var_list}
    self._save_theta0_ops = [
      tf.assign(theta0, theta) for theta, theta0 in self._theta_0.items()]
    self._reset_theta_ops = [
      tf.assign(theta, theta0) for theta, theta0 in self._theta_0.items()]

  def _calculate_gradient_stats(self):
    # Sanity check
    checker.check_type(th.train_set, DataSet)
    checker.check_positive_integer(th.de_batch_size)
    checker.check_positive_integer(th.de_num_steps)
    self.show_status('Calculating gradient stats on training set ...')

    grad_square = [tf.square(self._grads[var]) for var in self._var_list]
    fetches = grad_square
    if self._metric_quantity is not None:
      assert isinstance(self._metric_quantity, Quantity)
      fetches.append(self._metric_quantity.quantities)

    # Check train_set
    train_set = th.train_set
    if not isinstance(train_set, DataSet):
      raise TypeError('!! th.train_set must be an instance of DataSet but has'
                      ' type `{}`'.format(type(train_set)))
    # Truncate train set if necessary
    if th.de_max_batches > 0:
      if isinstance(train_set, SequenceSet): size = th.de_max_batches
      else: size = th.de_batch_size * th.de_num_steps * th.de_max_batches
      train_set = train_set[:size]
      train_set.name = 'train_set[:{}]'.format(size)
      # Show info
      # self.show_status('train_set truncated to de_max_batches({})'.format(size))
    # num_steps = th.eval_num_steps if th.eval_num_steps else th.de_num_steps
    num_steps = th.de_num_steps
    outputs = self._model.evaluate(
      fetches, train_set, batch_size=th.de_batch_size,
      num_steps=num_steps, verbose=True)

    # Show metric on training set if provided
    if self._metric_quantity is not None:
      metric_quantities = outputs.pop(-1)
      metric_val = self._metric_quantity.apply_np_summ_method(metric_quantities)
      console.supplement('{} on training set = {}'.format(
        self._metric_quantity.name,
        th.decimal_str(metric_val, th.val_decimals)))

    # Assign mean square grads
    assign_ops = []
    for var, output_list in zip(self._var_list, outputs):
      assert isinstance(output_list, list)
      mean_square = np.mean(output_list, axis=0)
      sqrt_mean_square = np.sqrt(mean_square)
      assign_ops.append(tf.assign(self._sqrt_MS_g[var], sqrt_mean_square))
    self._model.session.run(assign_ops)

    # After gradient stats have been calculated, save them into disk
    # .. if necessary
    if th.de_save_train_stats:
      th.train_stats_exists = True
      # When th.train_stats_exists is True,
      # .. saver will initiated with _sqrt_MS_g
      self._model.agent.reset_saver()
      self._model.agent.save_model(suffix='DeStat')
      self.show_status('sqrt_MS_g saved to checkpoint')

  # endregion : Private Methods

  # region : Public Methods

  def reset_parameters(self):
    self._model.session.run(self._reset_theta_ops)
    self.show_status('Model parameters have been reset.')

  def minimize(self, loss):
    assert isinstance(loss, tf.Tensor)
    grads = OrderedDict({
      v: g for v, g in zip(self._var_list, tf.gradients(loss, self._var_list))})
    self._grads = grads
    update_ops = []
    for var in self._var_list:
      new_var = (var - self._tf_eta * tf.divide(
        grads[var], self._sqrt_MS_g[var] + self._epsilon)
                 + self._decay_rate[var] * (self._theta_0[var] - var))
      update_ops.append(tf.assign(var, new_var))

    if not th.train_stats_exists or th.de_save_train_stats:
      self._calculate_gradient_stats()
    return tf.group(*update_ops)

  def set_hyper_parameters(self, eta, lambd):
    assert isinstance(eta, float) and isinstance(lambd, float)
    assert eta > 0 and lambd > 0
    self._eta, self._lambda = eta, lambd
    # Set eta and lambda to graph
    self._model.session.run(
      [self._set_eta_op, self._set_lambda_op],
      feed_dict={self._eta_placeholder: eta, self._lambda_placeholder: lambd})
    # Show status
    console.show_info('Hyper parameters for dynamic evaluation updated:')
    console.supplement('eta = {}, lambda = {}'.format(eta, lambd))

  # endregion : Public Methods


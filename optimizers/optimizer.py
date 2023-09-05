from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import console
from tframe import context
from tframe import hub as th
from tframe import pedia

from tframe.utils.stark import decayable


class Optimizer(object):
  """Tframe optimizer class allowing more sophisticated operations on gradients.
  For example, gradient clipping, and learning rate schedule.
  """

  def __init__(self, tf_optimizer):
    assert isinstance(tf_optimizer, tf.train.Optimizer)
    self.tf_optimizer = tf_optimizer

    self.reset_tf_optimizer = None


  def minimize(self, loss, var_list=None):
    # Step 1: compute gradients
    grads_and_vars = self._compute_gradients(loss, var_list=var_list)
    grads = [g for g, _ in grads_and_vars]

    # Step 2: apply gradients
    g_v_for_update = grads_and_vars
    if th.batchlet_size is not None:
      assert th.batchlet_size > 0

      if th.gradlet_in_device:
        gas = [
          tf.Variable(initial_value=tf.zeros_like(g), trainable=False,
                      shape=g.shape, dtype=g.dtype)
          for g, _ in grads_and_vars]
        coef = tf.placeholder(dtype=th.dtype, shape=())
        init_gas = tf.group(*[tf.assign(ga, tf.zeros_like(ga)) for ga in gas])
        assign_add_grads = tf.group(
          *[tf.assign_add(ga, coef * g) for g, ga in zip(grads, gas)])
        g_v_for_update = [(g, v) for (_, v), g in zip(grads_and_vars, gas)]
      else:
        gps = [tf.placeholder(g.dtype, g.shape) for g, _ in grads_and_vars]
        g_v_for_update = [(g, v) for (_, v), g in zip(grads_and_vars, gps)]

    update = self.tf_optimizer.apply_gradients(g_v_for_update)

    # Step 3: apply decoupled weight decay if required
    if th.decoupled_l2_penalty > 0:
      assert th.decoupled_l2_penalty < 1
      vars_to_decay = [v for _, v in grads_and_vars if decayable(v)]
      with tf.control_dependencies([update]):
        update_with_decay = tf.group(*[
          tf.assign_sub(v, v * th.decoupled_l2_penalty) for v in vars_to_decay])
      update = update_with_decay

    # Set reset_tf_optimizer if necessary
    if th.reset_optimizer_after_resurrection and th.lives > 0:
      self.reset_tf_optimizer = tf.variables_initializer(
        self.tf_optimizer.variables())

    # Return operators accordingly
    if th.batchlet_size is None:
      return update

    if th.gradlet_in_device:
      return coef, init_gas, assign_add_grads, update

    return grads, gps, update


  def _compute_gradients(self, loss, var_list=None):
    # Sanity check
    assert isinstance(loss, tf.Tensor)

    # Compute gradients using default method
    grads_and_vars = self.tf_optimizer.compute_gradients(
      loss, var_list=var_list)

    # Deal with Nan if required
    if th.opt_nan_protection: grads_and_vars = [
      (self.deal_with_nan(grad), var) for grad, var in grads_and_vars]

    # Modify learning rate after resurrection if necessary
    res_decay = th.opt_lr_multiplier
    if res_decay < 1.0:
      assert res_decay > 0
      grads_and_vars = [(grad * res_decay, var) for grad, var in grads_and_vars]

    # Clip gradients if necessary
    if th.clip_threshold > 0:
      grads_and_vars = self.clip_gradients(grads_and_vars)

    return grads_and_vars


  # region: Static and class methods

  @staticmethod
  def deal_with_nan(grad):
    # Reference: https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
    assert isinstance(grad, tf.Tensor)
    return tf.where(tf.is_nan(grad), tf.zeros(grad.shape, dtype=th.dtype),
                    grad)

  @staticmethod
  def clip_gradients(grads_and_vars):
    """This method was migrated from GradientClipOptimizer which has been
       deprecated"""
    clip_method = th.clip_method
    bound = th.clip_threshold
    assert clip_method in ('norm', 'value', 'global_norm', 'avg_norm')

    if clip_method in ('norm', 'value', 'avg_norm'):
      if clip_method == 'norm':
        method = lambda g: tf.clip_by_norm(g, bound)
      elif clip_method == 'value':
        method = lambda g: tf.clip_by_value(g, -bound, bound)
      else:
        method = lambda g: tf.clip_by_average_norm(g, bound)
      grads_and_vars = [(method(grad), var) for grad, var in grads_and_vars]
    else:
      assert clip_method == 'global_norm'
      grads = [g for g, _ in grads_and_vars]
      clipped_grads, _ = tf.clip_by_global_norm(grads, bound)
      vars_ = [v for _, v in grads_and_vars]
      grads_and_vars = list(zip(clipped_grads, vars_))

    return grads_and_vars

  @classmethod
  def get_optimizer(cls, identifier):
    """Get tframe optimizer from an identifier, which can be
       (1) a string or a class, e.g., `adam` or tf.train.Adam
       (2) an instance of tensorflow optimizer, e.g., tf.train.Adam()
       (3) a tframe optimizer
    """
    # Case (3) this branch is blocked for now
    if isinstance(identifier, Optimizer):
      assert False
      return identifier
    # Case (1)
    optimizer = cls.get_tf_optimizer(identifier)
    # Case (2) this branch will disable the automatic setting of lr decay,
    #   highly not recommended
    if not isinstance(optimizer, tf.train.Optimizer):
      raise TypeError('!! Failed to get optimizer')

    # Set tensorflow optimizer to context for future use
    context.tf_optimizer = optimizer
    # The code below is migrated from previous version
    if not th.save_train_opt_vars: optimizer._name = pedia.train_opt

    # Wrap and return
    return Optimizer(optimizer)

  @classmethod
  def get_tf_optimizer(cls, optimizer):
    if isinstance(optimizer, tf.train.Optimizer): return optimizer
    # Consider learning rate decay
    lr = cls.get_learning_rate()
    # Set modifier to learning rate
    if context.lr_coef is None:
      context.lr_coef = tf.Variable(1.0, trainable=False, name='lr_var_coef')
    lr = lr * context.lr_coef

    if optimizer in ['adam', tf.train.AdamOptimizer]:
      return tf.train.AdamOptimizer(
        learning_rate=lr, beta1=th.beta1, beta2=th.beta2,
        epsilon=th.optimizer_epsilon)
    elif optimizer in ['rmsprop', tf.train.RMSPropOptimizer]:
      return tf.train.RMSPropOptimizer(
        learning_rate=lr, momentum=th.momentum, epsilon=th.optimizer_epsilon)
    elif optimizer in ['sgd', tf.train.GradientDescentOptimizer]:
      return tf.train.GradientDescentOptimizer(learning_rate=lr)
    return optimizer(th.learning_rate)

  @classmethod
  def get_learning_rate(cls):
    if not th.lr_decay_enabled: return th.learning_rate
    # Initialize steps as variables
    if context.lr_global_step is None:
      assert context.lr_decay_steps is None
      context.lr_global_step = tf.Variable(
        0, trainable=False, dtype=tf.float32, name='lr_var_global_step')
      context.lr_decay_steps = tf.Variable(
        999999, trainable=False, dtype=tf.float32, name='lr_var_decay_steps')

    # Find method
    method = th.lr_decay_method.lower().replace('-', '_')
    lr = None
    if method in ('exp', 'exponential'): lr = None
    elif method in ('piece', 'piecewise'): lr = None
    elif method in ('poly', 'polynomial'): lr = None
    elif method in ('inverse', 'inverse_time'): lr = None
    elif method in ('cosine_restart',): lr = None
    elif method in ('linear_cosine',): lr = None
    elif method in ('noisy_linear_cosine',): lr = None
    elif method in ('cos', 'cosine'):
      lr = tf.train.cosine_decay(
        th.learning_rate, global_step=context.lr_global_step,
        decay_steps=context.lr_decay_steps, alpha=th.ending_lr)

    # Check and return
    if lr is None: raise KeyError(
      'Unknown learning rate decay method `{}`'.format(th.lr_decay_method))
    console.show_status(
      '`{}` learning rate decay has come into effect.'.format(method), '++')
    return lr

# endregion: Static and class methods


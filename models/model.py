from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import tf
from collections import OrderedDict

import tframe as tfr
from tframe import DataSet
from tframe import hub
from tframe import checker
from tframe import console
from tframe import context
from tframe import pedia

from tframe.utils.display.progress_bar import ProgressBar
from tframe.enums import InputTypes
from tframe.core import with_graph
from tframe.core import TensorSlot, NestedTensorSlot
from tframe.core import SummarySlot, OperationSlot, IndependentSummarySlot
from tframe.core import Group
from tframe.core.agent import Agent
from tframe.core.quantity import Quantity

from tframe.trainers.metric_slot import MetricSlot
from tframe.trainers.scheme import TrainScheme
from tframe.trainers.trainer import Trainer, TrainerHub
from tframe.trainers.smartrainer import SmartTrainer
from tframe.trainers.metrics_manager import MetricsManager

from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.bigdata import BigData
from tframe.data.perpetual_machine import PerpetualMachine


class Model(object):
  """
  Base class of [all?] kinds of models built on TensorFlow
  """
  model_name = 'default'

  def __init__(self, mark=None):
    # Model mark usually helps to decide the folder name
    # TODO: need to be refactored
    self.mark = hub.mark or mark
    assert mark is not None
    if hub.prefix is not None: self.mark = hub.prefix + self.mark
    if hub.suffix is not None: self.mark += hub.suffix
    if hub.script_suffix not in (None, ''):
      self.mark += '_Sc' + hub.script_suffix
    # TODO: set prune iteration number.
    #       At this time configs conflicts are not smoothed.
    pr = hub.pruning_rate
    if pr is not None and pr > 0:
      self.mark += '_pr{}'.format(hub.pruning_iterations)
    hub.mark = self.mark

    # Each model has an agent to deal with some tensorflow stuff
    self.agent = Agent(self)

    # Define slots
    # 2020-6-10 | William |
    #   outputs should be a Group which is more general for error injection
    #   tframe 2.0 should be using such way to describe a Model
    self._outputs = TensorSlot(self)
    self.val_outputs = TensorSlot(self)

    self._shadow_input = None

    # Compromising way to enable additional error injection
    self._forms_for_injection = []

    self._metrics_manager = MetricsManager(self)

    self._validation_summary = SummarySlot(self)
    self._batch_val_summ = IndependentSummarySlot(self, 'batch_metric_summ')

    self._loss = TensorSlot(self, 'Loss')
    self._train_step = OperationSlot(self, name='Train-step')
    self._train_step_summary = SummarySlot(self)


    self.validate_group = Group(
      self, self._validation_summary, name='Validate-group')

    if hub.batchlet_size is None:
      self._update_group = Group(
        self, self._loss, self._train_step, self._train_step_summary,
        name='Update-group')
    elif hub.gradlet_in_device:
      self._gas_coef = None
      self._init_gas = OperationSlot(self, name='Init-GAS')
      self._accum_gas = OperationSlot(self, name='Accumulate-Grads')
      self._update_group = Group(
        self, self._loss, self._accum_gas, name='Update-group')
    else:
      self._batchlet_grads = NestedTensorSlot(self, 'Batchlet-Gradients')
      self._gradient_placeholders = None
      self._update_group = Group(
        self, self._loss, self._batchlet_grads, name='Update-group')

    # Slots for exporting np values to note
    self.grads_slot = NestedTensorSlot(self, 'Gradients')
    self.general_tensor_slot = NestedTensorSlot(self, 'General-Tensor')

    # Private attributes
    self._default_net = None  # TODO to be removed
    self._optimizer = None
    self._built = False
    self._scheme = None

    self.shadows = OrderedDict()
    self._shadow_assign_group = None

    # Public attributes
    self.counter = None
    self.rounds = None
    self.launched = False

    # Quantities
    self.loss_quantity = None

  # region : Properties

  # region : Accessor

  @property
  def affix(self):
    return 'model'

  @property
  def graph(self):
    return self.agent.graph

  @property
  def session(self):
    return self.agent.session

  @property
  def metrics_manager(self):
    return self._metrics_manager

  @property
  def key_metric(self):
    if not self.metrics_manager.has_metric: return None
    return self.metrics_manager.early_stop_slot

  @property
  def eval_metric(self):
    if not self.metrics_manager.has_metric: return None
    return self.metrics_manager.eval_slot

  @property
  def outputs(self):
    assert isinstance(self._outputs, TensorSlot)
    return self._outputs

  @property
  def loss(self):
    assert isinstance(self._loss, TensorSlot)
    return self._loss

  @property
  def train_step(self):
    assert isinstance(self._train_step, OperationSlot)
    return self._train_step

  @property
  def built(self):
    assert isinstance(self._built, bool)
    return self._built

  @property
  def record(self):
    if not self.key_metric.activated: return None
    else: return self.key_metric.record

  @property
  def variable_to_save(self):
    """Should be called in with_graph decorator"""
    vars = (tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) +
            tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))
    # Remove `do not save` vars
    vars = [var for var in vars
            if var not in tf.get_collection(pedia.do_not_save)]

    filter_by_name = lambda key: [var for var in vars if key not in var.name]
    # Remove `train_opt` vars if necessary
    if not hub.save_train_opt_vars:
      vars = filter_by_name(pedia.train_opt)
      vars = filter_by_name('Optimizer')
    # Remove variables defined in optimizer
    vars = filter_by_name('lr_var')
    # Remove `dynamic_opt` vars
    vars = filter_by_name(pedia.dynamic_opt)
    # Krause optimizer related vars (TODO: need to be refactored)
    vars = filter_by_name('de_theta0')
    if not hub.train_stats_exists: vars = filter_by_name('de_sqrt_MS_g')
    return vars

  @property
  def metric_foreach(self):
    metrics = tf.get_collection(pedia.metric_foreach)
    assert len(metrics) == 1
    return metrics[0]

  @property
  def parameters_dict(self):
    # Fetch all trainable variables
    trainable_variables = tf.trainable_variables()
    values = self.session.run(trainable_variables)
    # Wrap them into a dictionary and return
    parameters = {}
    for t, v, in zip(trainable_variables, values):
      parameters[t.name] = v
    return parameters

  # endregion : Accessor

  # region : Properties to be overrode

  @property
  def description(self):
    return 'No description'

  @property
  def input_type(self):
    return InputTypes.BATCH
  
  # endregion : Properties to be overrode

  # endregion : Properties

  # region : Building

  @with_graph
  def build(self, **kwargs):

    # Smooth out flags before important actions
    hub.smooth_out_conflicts()

    # Initialize pruner if necessary
    if any([hub.prune_on, hub.weights_mask_on, hub.etch_on,
            hub.force_to_use_pruner]):
      # import here to prevent circular import (temporarily)
      from tframe.advanced.prune.pruner import Pruner
      tfr.context.pruner = Pruner(self)

    # Initialize puller (before init_shadow is called) if required
    if hub.cl_reg_on:
      from tframe.advanced.synapspring.puller import Puller
      tfr.context.puller = Puller(self)

    # If optimizer if not provided here, try hub.get_optimizer()
    #   this requires that th.optimizer and th.learning_rate have been provided
    if 'optimizer' not in kwargs: kwargs['optimizer'] = hub.get_optimizer()
    # Call successor's _build method
    self._build(**kwargs)
    # Initialize monitor
    self._init_monitor()
    # Set built flag
    self._built = True
    # Show build info
    console.show_status('Model built successfully:')
    self.agent.take_notes('Model built successfully')
    self.agent.take_notes('Structure:', date_time=False)
    # Description may be a model structure
    description = self.description
    if not isinstance(description, (tuple, list)): description = [description]
    for line in description:
      assert isinstance(line, str)
      console.supplement(line)
      self.agent.take_notes(line, date_time=False)

    # Add metric slot to update group
    batch_metric = kwargs.get('batch_metric', [])
    if batch_metric:
      if not isinstance(batch_metric, (tuple, list)):
        batch_metric = [batch_metric]
      for metric_str in batch_metric:
        assert isinstance(metric_str, str)
        metric_slot = self.metrics_manager.get_slot_by_name(metric_str)
        self._update_group.add(metric_slot)

    # Register eval_metric if provided
    eval_metric = kwargs.get('eval_metric', None)
    if eval_metric is not None:
      assert isinstance(eval_metric, str)
      self.metrics_manager.register_eval_slot(eval_metric)

  def _build(self, optimizer=None, **kwargs):
    """Abstract method, must be implemented in different models
       Usually touches tensorflow api directly and plug tf ops into tfr slots
    """
    raise  NotImplementedError('!! build method not implemented')

  def _init_monitor(self):
    pass
    # TODO
    # if tfr.monitor.activated: tfr.monitor.init_monitor(self)

  def _init_shadows(self):
    assert len(self.shadows) == 0
    if not hub.create_shadow_vars: return
    # Create shadows
    assign_slots = []
    with tf.name_scope('Shadows'):
      for v in tf.trainable_variables():
        name = v.name.split(':')[0] + '-shadow'
        shape = v.shape.as_list()
        shadow = tf.Variable(np.zeros(shape, dtype=np.float32),
                             trainable=False, name=name, shape=shape)
        self.shadows[v] = shadow
        # Create assigning ops
        slot = OperationSlot(self)
        slot.plug(tf.assign(shadow, v))
        assign_slots.append(slot)

    # OperationSlot
    self._shadow_assign_group = Group(self, *assign_slots, name='AssignShadow')

  @with_graph
  def _define_train_step(self, optimizer=None, var_list=None):
    """ TODO: should be modified for tframe.optimizer
        self._train_step will be plugged only here
    """
    if not self._loss.activated:
      raise AssertionError('!! loss has not been activated yet')
    with tf.name_scope('Optimizer'):
      optimizer = hub.get_optimizer(optimizer)
      if optimizer is None: return
      self._optimizer = optimizer
      self.set_train_step(var_list)

  def set_train_step(self, var_list=None):
    from tframe.optimizers.optimizer import Optimizer
    optimizer: Optimizer = self._optimizer

    # Get package from optimizer
    package = optimizer.minimize(self._loss.op, var_list=var_list)

    # Unpack package accordingly
    if hub.batchlet_size is None: update = package
    elif hub.gradlet_in_device:
      coef, init_gas, assign_add_grads, update = package
      self._gas_coef = coef
      self._init_gas.plug(init_gas)
      self._accum_gas.plug(assign_add_grads)
    else:
      grads, grad_placeholders, update = package
      # If batchlet size is provided, self._train_step will not be activated
      self._batchlet_grads.plug(grads)
      self._gradient_placeholders = grad_placeholders

    # Note that in batchlet mode, train_step is not included in update_group
    self._train_step.plug(update)

  def reset_optimizer(self):
    from tframe.optimizers.optimizer import Optimizer
    assert isinstance(self._optimizer, Optimizer)
    self.session.run(self._optimizer.reset_tf_optimizer)
    console.show_status('TensorFlow optimizer has been reset.')

  def _merge_summaries(self):
    train_step_summaries = tf.get_collection(pedia.train_step_summaries)
    validation_summaries = tf.get_collection(pedia.validation_summaries)
    if len(train_step_summaries) > 0:
      self._train_step_summary.plug(tf.summary.merge(train_step_summaries))
    if len(validation_summaries) > 0:
      self._validation_summary.plug(tf.summary.merge(validation_summaries))

  # endregion : Building

  # region : Training

  def pretrain(self, **kwargs):
    """Method run in early training process, should be overrode"""
    if self._scheme is not None:
      assert isinstance(self._scheme, TrainScheme)
      trial = self._scheme.dequeue()
      if trial is not None: trial.initialize(self)

  @with_graph
  def train(self, training_set, trainer_hub=None, validation_set=None,
            snapshot=None, probe=None, evaluate=None, terminator=None,
            test_set=None, **kwargs):
    if trainer_hub is None:
      trainer_class = SmartTrainer if hub.smart_train else Trainer
    else:
      if not isinstance(trainer_hub, TrainerHub):
        raise TypeError('!! Input hub must be an instance of TrainerHub')
      trainer_class = trainer_hub.trainer_class
    trainer = trainer_class(
      self, training_set=training_set, validation_set=validation_set,
      snapshot=snapshot, probe=probe, evaluate=evaluate, terminator=terminator,
      test_set=test_set)
    trainer.train(hub=trainer_hub, **kwargs)

  def update_model(self, data_batch, **kwargs):
    """Default model updating method, should be overrode"""
    if hub.batchlet_size is None:
      feed_dict = self._get_default_feed_dict(data_batch, is_training=True)
      results = self._update_group.run(feed_dict, data=data_batch)
    elif hub.gradlet_in_device:
      assert isinstance(data_batch, DataSet)
      bls = tfr.hub.batchlet_size

      # Set gradient accumulators to 0.0
      self._init_gas.run()

      results = {}
      # Note here set is_training to False to bypass random selection logic
      for batchlet in data_batch.gen_batches(bls, is_training=False):
        coef = batchlet.size / data_batch.size
        feed_dict = self._get_default_feed_dict(batchlet, is_training=True)
        feed_dict[self._gas_coef] = coef

        # Calculate quantities and grads
        res_let = self._update_group.run(feed_dict, data=data_batch)

        # Accumulate quantities
        if len(results) == 0: results = {k: 0.0 for k in res_let.keys()}
        for k in res_let.keys(): results[k] += coef * res_let[k]

      # Run train step to update model
      self._train_step.run()
    else:
      # TODO: use batchlet to support infinite batch_size
      assert isinstance(data_batch, DataSet)
      bls = tfr.hub.batchlet_size
      results, gradients = {}, []

      # Note here set is_training to False to bypass random selection logic
      for batchlet in data_batch.gen_batches(bls, is_training=False):
        coef = batchlet.size / data_batch.size
        feed_dict = self._get_default_feed_dict(batchlet, is_training=True)

        # Calculate quantities and grads
        res_let = self._update_group.run(feed_dict, data=data_batch)
        grads_lets = res_let.pop(self._batchlet_grads)

        # Accumulate quantities
        if len(results) == 0: results = {k: 0.0 for k in res_let.keys()}
        for k in res_let.keys(): results[k] += coef * res_let[k]

        # Accumulate gradients
        if len(gradients) == 0: gradients = [0.0 for _ in grads_lets]
        gradients = [grad + gl * coef
                     for gl, grad in zip(grads_lets, gradients)]

      # Calculate gradients and update model
      self._train_step.run({
        p: g for p, g in zip(self._gradient_placeholders, gradients)})

    # Clip weights if necessary
    self._clip_weights()

    return results

  def get_data_batches(self, data_set, batch_size, num_steps=None,
                       shuffle=False, is_training=False):
    """ Get batch generator. This method is used both in training and
        evaluation/validation.

        It's trivial for FNN models. However, for RNN models, data_set may be
        (1) a SequenceSet in which the feature is a list of numpy arrays.
            each represents a sequence and the lengths may vary.
            e.g.
            data_set.feature = [
               xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,
               xxxxxxxxxxxxxxxxxxx,
               xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,
               xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,
            ], in which x represents a data point.
            In this case, batch size can be
            (a) 1 (by default)
            (b) larger than 1, active_len mechanism will be used,
                val_num_steps will be forced to -1
        (2) a DataSet consists of a single sequence, shape = [seq_len, *dim]
            In this case, batch size can be any integer

    :param data_set: an instance of DataSet or BigData from which data batches
                      will be extracted
    :param batch_size: if is None, default value will be assigned according to
                        the input type of this model
    :param num_steps: step number for RNN data batches
    :param shuffle: whether to shuffle
    :return: a generator or a list
    """
    # Data set must be an instance of DataSet or BigData
    assert isinstance(data_set, (DataSet, BigData, PerpetualMachine))

    if self.input_type is InputTypes.BATCH:
      # 1. For FNN, `num_steps` will be ignored, default batch_size is -1 (all)

      # If batch size is not specified and data is a DataSet, feed it all at
      #  once into model
      if batch_size is None and isinstance(data_set, DataSet):
        return [data_set.stack]

      # Otherwise batch_size must be an positive integer
      checker.check_positive_integer(batch_size)
      data_batches = data_set.gen_batches(
        batch_size, shuffle=shuffle, is_training=is_training)

    elif self.input_type is InputTypes.RNN_BATCH:
      # 2. For RNN, default batch_size is 1, default num_steps is -1 (all)
      #
      if num_steps is None: num_steps = -1
      if batch_size is None: batch_size = 1
      if batch_size < 0: batch_size = data_set.size

      # Cases:
      # (1) data_set is a DataSet but not a SequenceSet
      #     each data entry in data_dict will be considered as a consecutive
      #     sequence. batch_size and num_steps can be any integer
      # (2) data_set is a SequenceSet
      #     ---------------+------------------------+--------------------------
      #                    | num_steps = -1         | num_steps != -1
      #     ---------------+------------------------+--------------------------
      #                    |                        |
      #     batch_size = 1 | legal for all          | legal for all *
      #                    |                        |
      #     ---------------+------------------------+--------------------------
      #                    | train: legal for equal-length sequences since
      #                    |        act_len logic has not been implemented
      #     batch_size > 1 |        for training *
      #                    +------------------------+--------------------------
      #                    | val: legal for all     | TODO: not supported
      #     ---------------+------------------------+--------------------------
      #                                             | * n_to_one must be False

      # Check batch_size
      # it's legal for common DataSet to have num_steps > 0 while batch_size > 1
      checker.check_positive_integer(batch_size)
      if batch_size > 1 and isinstance(data_set, SequenceSet):
        #assert num_steps < 0  # XXXXXXXX
        # The constraint below is not necessary due to gather_indices mechanism
        # if is_training and not hub.use_gather_indices:
        #   assert data_set.equal_length
        pass

      # Check num_steps
      checker.check_type(num_steps, int)
      if num_steps != -1:
        # partition logic for n_to_one task has not been implemented yet
        assert not data_set.n_to_one

      # Generate batches
      data_batches = data_set.gen_rnn_batches(
        batch_size, num_steps, shuffle, is_training=is_training)
    else: raise ValueError('!! Can not resolve input type of this model')

    return data_batches

  def validate_model(self, data_set, batch_size=None, allow_sum=False,
                     verbose=False, seq_detail=False, num_steps=None):
    """Evaluate quantities in validate group of this model
    :param data_set: a tframe DataSet
    :param batch_size: if is None or -1, batch_size will be data_set.size
    :param allow_sum: whether to add tensorflow summaries TODO: to be deprecated
    :return: a dictionary in which keys are slots (may include loss and metric)
             and values are scalars corresponding to these slots
    """
    # Sanity check
    assert isinstance(data_set, DataSet)
    if data_set.is_rnn_input and num_steps is None:
      num_steps = hub.val_num_steps

    # - One-shot validation
    one_shot = False
    batch_is_all = batch_size in (-1, None) or batch_size == data_set.size
    # .. check one-shot qualification
    # .. .. the code below should be encapsulated
    if self.input_type is InputTypes.BATCH:
      # (1) e.g. small model on MNIST, CIFAR-10
      if batch_is_all: one_shot = True
    elif self.input_type is InputTypes.RNN_BATCH:
      # (2)
      if isinstance(data_set, SequenceSet):
        # (2-a)
        if batch_is_all and num_steps == -1 and data_set.equal_length:
          # e.g. AP, TO
          one_shot = True
      else:
        # (2-b)
        assert isinstance(data_set, DataSet)
        # assert batch_size in (1, -1, None)  # TODO
        # e.g. small model on WHB
        if num_steps == -1: one_shot = True

    # [Branch 1/2] do one-shot validation if is qualified
    # .. for RNN models, reset_batch flag of data_set should be set
    if one_shot:
      data_set = self._sanity_check_before_use(data_set)
      if self.input_type is InputTypes.RNN_BATCH:
        data_set.should_reset_state = True
      feed_dict = self._get_default_feed_dict(data_set, is_training=False)
      return self.validate_group.run(feed_dict, allow_sum=allow_sum,
                                     data=data_set)

    # [Branch 2/2] Otherwise do batch validation
    tensor_slots = self.validate_group.tensor_slots
    quantity_defs = [s.quantity_definition for s in tensor_slots]
    fetches = [q.quantities for q in quantity_defs]
    values = self.evaluate(
      fetches, data_set, batch_size, verbose=verbose, num_steps=num_steps)
    result_dict = OrderedDict()

    for val, qd, slot in zip(values, quantity_defs, tensor_slots):
      # Sanity check
      assert isinstance(qd, Quantity)
      if self.input_type is InputTypes.BATCH:
        assert isinstance(val, np.ndarray) and len(val) > 0
      else:
        assert isinstance(val, list)
        if not data_set.n_to_one:
          checker.check_type(val, np.ndarray)
      # Apply post-processor if provided
      if callable(slot.post_processor):
        val = slot.post_processor(val, data_set)
      # Apply np_summ_method on val
      scalar = qd.apply_np_summ_method(val, seq_detail)
      # Add summ to results
      result_dict[slot] = scalar

    return result_dict

  def take_down_metric(self, is_online):
    for metric in self.metrics_manager.metrics:
      assert isinstance(metric, MetricSlot)
      if not metric.activated: continue
      notes = 'Best {}: {:.3f}'.format(metric.symbol, metric.record)
      # if not is_online:
      #   notes += ', Best {} = {:.3f}'.format(metric.symbol, metric.mean_record)
      self.agent.take_notes(notes, date_time=False)

      # Add history into notes if necessary
      if hub.show_record_history_in_note:
        self.agent.take_notes(
          metric.metric_mean_history_str, date_time=False)
      # Add record and mean record to notes
      self.agent.put_down_criterion(
        'Best {}'.format(metric.symbol), metric.record)
      # if not is_online:
      #   self.agent.put_down_criterion('Best E({})', metric.mean_record)

    # Take down improvement if necessary
    if not hub.overwrite and hub.save_records:
      # Improvement is for smart booster. For incremental learning tasks,
      #  this criterion makes not sense.
      self.agent.put_down_criterion(
        'Improvement', self.metrics_manager.early_stop_slot.improvement)

  def end_round(self, rnd):
    self.key_metric.end_round(rnd)

  def bust(self, rnd):
    if self._scheme is not None:
      assert isinstance(self._scheme, TrainScheme)
      trial = self._scheme.dequeue()
      if trial is not None:
        trial.initialize(self)
        return False
      else: return True
    return True

  # endregion : Training

  # region : Public Methods

  def synchronize_shadow(self):
    self._shadow_assign_group.run()
    console.show_status('Shadow synchronized.')

  def handle_structure_detail(self):
    detail, total_params, dense_total = '', 0, 0
    d_t_d = getattr(self, 'structure_detail', None)
    if d_t_d: detail, total_params, dense_total = d_t_d
    # if hasattr(self, 'structure_detail'):
    #   detail, total_params, dense_total = self.structure_detail

    # Maybe take some notes
    params_str = 'Total params: {}'.format(total_params)
    hub.total_params = int(total_params)
    if hub.prune_on:
      hub.dense_total_params = dense_total
      hub.weights_fraction = 100.0 * total_params / dense_total
      params_str += ' ({:.2f}%)'.format(hub.weights_fraction)
    self.agent.take_notes(params_str)

    if hub.show_structure_detail:
      print('.. Structure detail:\n{}'.format(detail))

    if hub.export_structure_detail:
      self.agent.take_notes('Structure detail:', False)
      self.agent.take_notes(detail, False)

  def get_trainable_variables(self, f=None):
    if f is None: f = lambda _: True
    variables = [v for v in tf.trainable_variables() if f(v)]
    values = self.session.run(variables)
    variable_dict = OrderedDict()
    for t, v in zip(variables, values):
      variable_dict[t.name] = v
    return variable_dict

  def tune_lr(self, new_lr=None, coef=1.0):
    assert False
    #TODO
    if self._optimizer is None:
      raise ValueError('!! Optimizer not defined yet')
    if self._optimizer.__class__ in [tf.train.AdamOptimizer]:
      lr_name = '_lr'
    elif self._optimizer.__class__ in [tf.train.GradientDescentOptimizer]:
      lr_name = '_learning_rate'
    else:
      raise TypeError('!! Unsupported optimizer for lr tuning')

    old_lr = self._optimizer.__getattribute__(lr_name)
    if new_lr is None: new_lr = old_lr * coef
    self._optimizer.__setattr__(lr_name, new_lr)

    # Show status
    console.show_status(
      'Learning rate updated: {:.2e} => {:.2e}'.format(old_lr, new_lr))

    return new_lr

  def set_scheme(self, scheme):
    if not isinstance(scheme, TrainScheme):
      raise TypeError('!! scheme must be an instance of TrainScheme')
    self._scheme = scheme

  def shutdown(self):
    self.agent.shutdown()

  def launch_model(self, overwrite=False):
    results = self.agent.launch_model(overwrite)
    # Clip weights if necessary
    self._clip_weights()
    return results

  def evaluate(self, fetches, data, batch_size=None, postprocessor=None,
               verbose=False, num_steps=None, suppress_n_to_one=False):
    """
    Evaluate tensors based on data
    TODO: note that if num_steps != -1, outputs from a same sequence may be
          partitioned. e.g., if single_fetch, outputs will be
          [array_1_1, ..., array_1_k1, array_2_1, ..., array_2_k2, ...]
         |-------- input_1 ----------|------------ input_2 ----------|
         it's OK for seq2seq validation, but need to be post-proceeded in
         tasks like sequence classification (currently forbidden)

    :param fetches: a (tuple/list of) tf.Tensor(s) to be evaluated
    :param data: data used for evaluation
    :param batch_size: if not specified (None by default), batch_size will be
                       assigned accordingly. If assigned with a positive
                       integer, evaluation will be performed batch by batch.
    :param postprocessor: post-processor for outputs
    :return: commonly a (list of) tf.Tensor(s), each of which has the
             same batch size with the provided data
    """
    # Sanity check for fetches
    checker.check_fetchable(fetches)
    single_fetch = not isinstance(fetches, (tuple, list))
    # Wrap fetches into a list if necessary
    if single_fetch: fetches = [fetches]
    if data.is_rnn_input and num_steps is None: num_steps = hub.val_num_steps
    if batch_size is None: batch_size = data.size

    # Get outputs (sometimes fetches may contain operations which yields None)
    outputs = [[] for op in fetches if not isinstance(op, tf.Operation)]

    if verbose:
      bar = ProgressBar(data.get_round_length(batch_size, num_steps))
      console.show_status('Evaluating on {} ...'.format(data.name))

    for cursor, data_batch in enumerate(self.get_data_batches(
        data, batch_size, num_steps)):
      data_batch = self._sanity_check_before_use(data_batch)
      # Get batch outputs          fetches[0]  fetches[1]
      #  for FNN, batch_outputs = [np_array_1, np_array_2, ...]
      #           each np_array_k have a same batch_size
      #  for RNN, batch_outputs = [[s1_1, s1_2, ..., s1_N],       <= fetches[0]
      #                            [s2_1, s2_2, ..., s2_N], ...]  <= fetches[1]
      #           N is the batch_size, and each sk_i is a numpy array
      batch_outputs = self._evaluate_batch(
        fetches, data_batch, num_steps=num_steps,
        suppress_n_to_one=suppress_n_to_one)
      assert isinstance(batch_outputs, list)
      assert len(batch_outputs) == len(outputs)

      # Add batch_outputs to outputs accordingly
      for i, batch_output in enumerate(batch_outputs):
        assert isinstance(outputs[i], list)
        output_is_a_batch = fetches[i].shape.as_list()[0] is None
        if self.input_type is InputTypes.RNN_BATCH and output_is_a_batch:
          # batch_output is [s1_1, s1_2, ..., s1_N]
          assert isinstance(batch_output, list)
          outputs[i] = outputs[i] + batch_output
        else:
          # batch_output is a numpy array of length batch_size
          outputs[i].append(batch_output)

      # Show progress bar if necessary
      if verbose: bar.show(cursor + 1)

    # Merge outputs if necessary
    if self.input_type is InputTypes.BATCH:
      outputs = [np.concatenate(array_list, axis=0) for array_list in outputs]

    # Post-proceed and return
    if postprocessor is not None:
      assert callable(postprocessor)
      outputs = postprocessor(outputs)

    assert isinstance(outputs, list)
    if single_fetch: outputs = outputs[0]
    return outputs

  def rehearse(self, path=None, export_graph=False, build_model=True,
               mark=None):
    """This method build and launch model, show structure detail and export
    tensorflow logs containing graph which can be visualized in TensorBoard."""
    import os, sys
    from tframe import hub as th

    if path is None: path = os.path.join(sys.path[0], 'tmp')
    th.summary = export_graph
    th.job_dir = path

    if mark is not None: self.mark = mark
    if build_model: self.build(optimizer=tf.train.GradientDescentOptimizer(0.0))
    self.launch_model(overwrite=True)

  # endregion : Public Methods

  # region : Private Methods

  def _clip_weights(self):
    if hub.clip_weight_at is None: return
    from tframe.nets.net import Net
    assert isinstance(self, Net) and isinstance(self, Model)
    self.session.run(self.weight_clip_ops)

  def _evaluate_batch(self, fetch_list, data_set, **kwargs):
    raise NotImplementedError

  @with_graph
  def _get_default_feed_dict(self, batch, is_training):
    feed_dict = {}
    default_feed_collection = tf.get_collection(pedia.default_feed_dict)

    # Handle conflict caused by non_train_input
    input_key, target_key = 'input', 'targets'
    non_train_cond_triggered = all(
      [not is_training, hub.non_train_input_shape is not None])
    if non_train_cond_triggered:
      input_key, target_key = pedia.non_train_input, pedia.non_train_target

    for tensor in default_feed_collection:
      # Get tensor name
      name: str = tensor.name.split('/')[-1].split(':')[0]

      if input_key == name.lower():
        feed_dict[tensor] = batch[pedia.features]
      elif name == target_key:
      # elif tensor.name.lower() in (target_key,):
      # elif target_key in tensor.name:
        # TODO: when predict without outputting loss ...
        if batch.targets is not None: feed_dict[tensor] = batch.targets
      elif pedia.gather_indices in tensor.name:
        # TODO: when batch.size is 1, gather_indices is not necessary
        #       However, Quantity will never know the exact batch size
        feed_dict[tensor] = batch.gather_indices
      else:
        # TODO: use this ugly patch to circumvent non-train input issue
        if non_train_cond_triggered and name == 'targets': continue
        val = batch.data_dict.get(name, None)
        if val is None: val = batch.properties.get(name, None)
        if val is not None: feed_dict[tensor] = val

    feed_dict.update(self.agent.get_status_feed_dict(is_training))

    # This method is created for blind_denoise/quan.py
    # Each func below should have the signature below:
    #   def func(batch: DataSet) -> dict:
    for func in context.feed_dict_fillers: feed_dict.update(func(batch))

    return feed_dict

  def _sanity_check_before_use(self, data):
    # Make sure data is legal
    if not isinstance(data, DataSet):
      raise TypeError('!! Input data must be an instance of DataSet')
    # Make sure model has been built
    if not self.built: raise ValueError('!! Model not built yet')
    # Make sure model has been launched
    if not self.launched: self.launch_model(overwrite=False)
    # Make sure data type matches model input type
    if self.input_type is InputTypes.RNN_BATCH: data = data.as_rnn_batch
    else: assert not data.is_rnn_input
    return data

  # endregion : Private Methods


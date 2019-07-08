from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import hub as th
from tframe import checker
from tframe import console
from tframe.models.model import Model
from tframe.data.dataset import DataSet
from tframe.core.quantity import Quantity

from tframe.advanced.krause_evaluator import KrauseEvaluator


class DynamicEvaluator(object):
  """Currently only krause evaluator is used.
     How to setup appropriately:
     (1) set th.train_set if not th.train_stats_exists
     (2) optional:
         (2.1) set th.de_val_pct to specify length of val_set for HP searching
         (2.2) set th.de_max_batches to specify max batches used for calculating
               gradient statistics
         (2.3) specify th.de_eta and th.de_lambda to fix corresponding HP

     TODO: ISSUES TO FIX:
     CUDA_ERROR_OUT_OF_MEMORY occurs during calculating gradstats

  """

  def __init__(self, model):
    assert isinstance(model, Model)
    self.model = model

    # Preparation
    self._loss_tensor = self.model.loss.op
    self._quantities_op = None
    self._quantity = None
    self._get_quantity()

    self.optimizer = KrauseEvaluator(
      model, metric_quantities=self._quantities_op)
    self._train_op = self.optimizer.minimize(self._loss_tensor)

  @property
  def _dynamic_fetches(self):
    assert self._train_op is not None and self._quantities_op is not None
    return [self._quantities_op, self._train_op]

  def evaluate(self, data_set, val_set=None):
    """Val set is for hyper-parameters tuning
    """
    assert isinstance(data_set, DataSet)
    assert isinstance(val_set, DataSet) or val_set is None

    # Do hyper-parameter search if necessary
    if val_set is not None and (th.de_eta is None or th.de_lambda is None):
      self._search_hp(val_set)

    # Dynamic evaluation on test set
    assert th.de_eta > 0 and 0 < th.de_lambda < 1
    self._dynamic_eval(data_set, th.de_eta, th.de_lambda)


  def _dynamic_eval(self, data_set, lr, lambd, prompt='[Dynamic Evaluation]'):
    assert isinstance(data_set, DataSet)
    console.show_status('lr = {}, lambd = {}'.format(lr, lambd), prompt)
    # Reset parameters
    self.optimizer.reset_parameters()
    # Set HP to optimizer
    self.optimizer.set_hyper_parameters(lr, lambd)
    # Do dynamic evaluation
    output = self.model.evaluate(
      self._dynamic_fetches, data_set, batch_size=1, verbose=True,
      num_steps=th.de_num_steps)[0]
    assert isinstance(self._quantity, Quantity)
    metric = self._quantity.apply_np_summ_method(output)
    console.supplement('Dynamic metric = {:.3f}'.format(metric))
    return metric

  def _search_hp(self, val_set):
    assert isinstance(val_set, DataSet)
    # Truncate val_set
    size = int(val_set.size * th.de_val_pct)
    val_set = val_set[:size]
    # Validate on val_set
    self._validate(val_set)
    # Generate HP grid
    lr_list = self.optimizer.lrlist if th.de_eta is None else [th.de_eta]
    lambd_list = (self.optimizer.lamblist if th.de_lambda is None
                  else [th.de_lambda])
    hp_grid = [(lr, lambd) for lr in lr_list for lambd in lambd_list]
    # Do grid search
    console.show_status('Searching hyper-parameters on validation set ...')
    best_metric = None
    for i, (lr, lambd) in enumerate(hp_grid):
      metric = self._dynamic_eval(
        val_set, lr, lambd, '[{}/{}]'.format(i + 1, len(hp_grid)))
      if best_metric is None or metric < best_metric:
        best_metric = metric
        th.de_eta, th.de_lambda = lr, lambd

  # def _reset_train_op(self, lr):
  #   assert isinstance(self._loss_tensor, tf.Tensor)
  #   optimizer = self.opt_def(lr, name=pedia.dynamic_opt)
  #   self._train_op = optimizer.minimize(self._loss_tensor)
  #   self.model.session.run(tf.variables_initializer(optimizer.variables()))

  def _get_quantity(self):
    tensor_slots = self.model.validate_group.tensor_slots
    metric_quantity = [s.quantity_definition for s in tensor_slots][0]
    assert isinstance(metric_quantity, Quantity)
    quantities_op = metric_quantity.quantities
    self._quantity = metric_quantity
    self._quantities_op = quantities_op

  def _validate(self, data_set, batch_size=1, num_steps=1000):
    result_dict = self.model.validate_model(
      data_set, batch_size=batch_size, verbose=True, num_steps=num_steps,
      seq_detail=th.val_info_splits > 0)
    val = list(result_dict.values())[0]
    console.supplement('Metric = {:.3f}'.format(val))

  @staticmethod
  def dynamic_evaluate(model, data_set, val_set=None):
    de = DynamicEvaluator(model)
    de.evaluate(data_set, val_set)



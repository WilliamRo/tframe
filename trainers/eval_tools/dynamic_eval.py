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
     (3) set de_num_steps. For character level model it should be approx 20.
         For word level model, it should be approx 5. (according to krause18)

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
    lr_option, dc_option = th.de_eta_option, th.de_lambda_option
    assert isinstance(lr_option, list) and isinstance(lr_option, list)
    hp_grid = [(lr, dc) for lr in lr_option for dc in dc_option]
    if val_set is not None and len(hp_grid) > 1:
      lr, dc = self._search_hp(val_set, hp_grid)
    else:
      if len(hp_grid) != 1: raise ValueError(
        '!! HP searching is required yet val_set is not provided')
      lr, dc = hp_grid[0]

    # Dynamic evaluation on test set
    assert lr > 0 and 0 < dc < 1
    self._dynamic_eval(data_set, lr, dc)

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

  def _search_hp(self, val_set, hp_grid):
    assert isinstance(val_set, DataSet)
    # Truncate val_set
    size = int(val_set.size * th.de_val_pct)
    val_set = val_set[:size]
    # Validate on val_set
    self._validate(val_set)
    # Do grid search
    console.show_status('Searching hyper-parameters on validation set ...')
    best_metric, best_lr, best_dc = None, None, None
    for i, (lr, lambd) in enumerate(hp_grid):
      metric = self._dynamic_eval(
        val_set, lr, lambd, '[{}/{}]'.format(i + 1, len(hp_grid)))
      if best_metric is None or metric < best_metric:
        best_metric = metric
        best_lr, best_dc = lr, lambd
    return best_lr, best_dc

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



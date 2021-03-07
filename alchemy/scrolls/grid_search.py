from collections import OrderedDict

from tframe import console

from .scroll_base import Scroll
from ..hyper_param import CategoricalHP


class GridSearch(Scroll):

  name = 'Grid Search'
  valid_HP_types = (CategoricalHP,)

  def __init__(self, hyper_params, constraints, times=1, **kwargs):
    # Call parent's constructor
    super(GridSearch, self).__init__(hyper_params, constraints, **kwargs)
    # Specific variables
    assert isinstance(times, int) and times > 0
    self.times = times

  def hp_descartes(self, hyper_params=None):
    """Provide a generator of hyper-parameters for grid search"""
    if hyper_params is None: hyper_params = list(self.hyper_params.values())
    if len(hyper_params) == 0: yield OrderedDict()
    else:
      first_hp = hyper_params[0]
      assert isinstance(first_hp, CategoricalHP)
      for val in first_hp.choices:
        configs = OrderedDict()
        configs[first_hp.name] = val
        for cfg_dict in self.hp_descartes(hyper_params[1:]):
          configs.update(cfg_dict)
          yield configs

  def combinations(self):
    for run_id in range(self.times):
      history = []
      for configs in self.hp_descartes():
        assert isinstance(configs, OrderedDict)
        self.apply_constraint(configs)
        # Check history
        config_hash = hash(tuple(sorted([(k, v) for k, v in configs.items()])))
        if config_hash in history: continue
        history.append(config_hash)
        # Show status
        console.show_status('', '[# {} of Run {}/{}]'.format(
          len(history), run_id + 1, self.times))

        yield configs


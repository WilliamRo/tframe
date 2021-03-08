import numpy as np
from collections import OrderedDict

from tframe import console

from ..hyper_param import HyperParameter


class Scroll(object):

  name = 'Scroll'
  valid_HP_types = ()

  enable_hp_types = False
  logging_is_needed = False

  def __init__(self, hyper_params, constraints, observation_fetcher=None,
               greater_is_better=None, **kwargs):
    # Set hyper-parameters
    self.hyper_params = OrderedDict()
    self.set_search_space(hyper_params)
    # Set constraint
    self.constraints = OrderedDict()
    self.set_constraints(constraints)
    # Set observation_fetcher
    self._observation_fetcher = observation_fetcher
    self._greater_is_better = greater_is_better
    # Save key word arguments
    self.kwargs = kwargs
    # Buffer
    self.seen_previously = []
    self.log_strings = []

  # region: Properties

  @property
  def details(self):
    return self.name

  @property
  def best_criterion(self):
    ys = [t[1] for t in self.seen_previously]
    return sorted(ys)[-1 if self.greater_is_better else 0]

  @property
  def observation_fetcher(self):
    if not callable(self._observation_fetcher): raise AssertionError(
      '!! The observation_fetcher has not been appropriately set.')
    return self._observation_fetcher

  @property
  def greater_is_better(self):
    if not isinstance(self._greater_is_better, bool): raise AssertionError(
      '!! Whether greater criterion is better is unknown.')
    return self._greater_is_better

  # endregion: Properties

  def combinations(self):
    """A generator emitting hyper-parameter combinations.
     Each combination emitted may base on some state of the scroll. """
    raise NotImplementedError

  def set_search_space(self, hyper_params):
    assert isinstance(hyper_params, list)
    if len(hyper_params) == 0: return
    # Find appropriate HP type if different types are allowed
    if self.enable_hp_types:
      hyper_params = [hp.seek_myself() for hp in hyper_params]
    # Show hyper-parameters setting
    console.show_info('Hyper Parameters -> {}'.format(self.name))
    for hp in hyper_params:
      assert isinstance(hp, HyperParameter)
      assert isinstance(hp, self.valid_HP_types)
      console.supplement('{}: {}'.format(hp.name, hp.option_str), level=2)
      self.hyper_params[hp.name] = hp

  def set_constraints(self, constraints):
    assert isinstance(constraints, dict)
    if len(constraints) == 0: return
    # Show constraints
    console.show_info('Constraints')
    for cond, cons in constraints.items():
      assert isinstance(cond, tuple) and isinstance(cons, dict)
      # Make sure each value in conditions with multi-value has been registered
      # Otherwise ambiguity will arise during constraint applying
      for k, v in cons.items():
        if isinstance(v, (tuple, set, list)):
          for choice in v:
            if choice not in self.hyper_params[k].choices:
              raise AssertionError(
                '!! Value {} for {} when {} should be registered first.'.format(
                  choice, k, cond))

      # Consider to skip some constraint settings here, i.e., when condition
      #   will never be satisfied

      console.supplement('{}: {}'.format(cond, cons), level=2)
      self.constraints[cond] = cons

  def apply_constraint(self, configs):
    assert isinstance(configs, OrderedDict)

    def _satisfy(condition):
      assert isinstance(condition, tuple) and len(condition) > 0
      for k, v in condition:
        if not isinstance(v, (tuple, list, set)): v = v,
        if k not in configs.keys(): return False
        elif configs[k] not in v: return False
      return True

    for cond, cons in self.constraints.items():
      assert isinstance(cons, dict)
      if _satisfy(cond):
        for key, val in cons.items():
          if isinstance(val, (set, tuple, list)):
            assert configs[key] in val
          else: configs[key] = val

  def is_better(self, a, b):
    if self.greater_is_better: return a > b
    return a < b

  def get_new_x_y(self, key_format='value'):
    # Check input
    assert key_format in ('value', 'dict', 'hp_dict', 'hyper_parameter_dict')
    # Get new observations
    new_observations = [
      ob for ob in self.observation_fetcher() if ob not in self.seen_previously]
    # Extent observation buffer
    self.seen_previously.extend(new_observations)
    # Get new_x_y
    if key_format != 'value': return new_observations
    return [(list(hp_dict.values()), c) for hp_dict, c in new_observations]

  def log(self, s):
    assert isinstance(s, str)
    self.log_strings.append(s)

  def get_log(self, empty_buffer=True):
    logs = self.log_strings
    if empty_buffer: self.log_strings = []
    return logs

  def _value_list_to_config(self, values):
    assert isinstance(values, (tuple, list))
    od = OrderedDict()
    for k, v in zip(self.hyper_params.keys(), values): od[k] = v
    return od




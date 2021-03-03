import numpy as np
from collections import OrderedDict

from tframe import console

from ..hyper_param import HyperParameter


class Scroll(object):

  name = 'Scroll'
  valid_HP_types = ()

  def __init__(self, hyper_params, constraints):
    # Set hyper-parameters
    self.hyper_params = OrderedDict()
    self.set_search_space(hyper_params)
    # Set constraint
    self.constraints = OrderedDict()
    self.set_constraints(constraints)

  def combinations(self):
    """A generator emitting hyper-parameter combinations.
     Each combination emitted may base on some state of the scroll. """
    raise NotImplementedError

  def set_search_space(self, hyper_params):
    assert isinstance(hyper_params, list)
    if len(hyper_params) == 0: return
    # Show hyper-parameters setting
    console.show_info('Hyper Parameters')
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

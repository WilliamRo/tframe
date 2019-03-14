from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import call
from collections import OrderedDict

from tframe import checker
from tframe.configs.flag import Flag
from tframe.trainers import SmartTrainerHub


flags, flag_names = None, None

def register_flags(config_class):
  global flags, flag_names
  flags = [attr for attr in
           [getattr(config_class, key) for key in dir(config_class)]
           if isinstance(attr, Flag)]
  flag_names = [f.name for f in flags]

register_flags(SmartTrainerHub)


def check_flag_name(method):
  def wrapper(obj, flag_name, *args, **kwargs):
    assert isinstance(obj, Helper)
    # Make sure flag_name is not in parameter list of obj
    if flag_name in obj.param_keys:
      raise ValueError('!! Key `{}` has already been set'.format(flag_name))
    # Make sure flag_name is registered by tframe.Config
    if flag_name not in flag_names:
      print(
        ' ! `{}` may be an invalid flag, press [Enter] to continue ...'.format(
          flag_name))
      input()
    method(obj, flag_name, *args, **kwargs)
  return wrapper


class Helper(object):

  # Class variables
  true_and_false = (True, False)
  true = True
  false = False

  def __init__(self, module_name):
    self.module_name = module_name
    self._check_module()

    self.common_parameters = OrderedDict()
    self.hyper_parameters = OrderedDict()
    self.constraints = OrderedDict()

  # region : Properties

  @property
  def command_head(self):
    return ['python', self.module_name] + [
      self._get_config_string(k, v) for k, v in self.common_parameters.items()]

  @property
  def param_keys(self):
    # Add keys from hyper-parameters
    keys = list(self.hyper_parameters.keys())
    # Add keys from common-parameters
    keys += list(self.common_parameters.keys())
    return keys

  # endregion : Properties

  # region : Public Methods

  def constrain(self, conditions, constraints):
    checker.check_type(conditions, dict)
    checker.check_type(constraints, dict)
    key = tuple([(k, v) for k, v in conditions.items()])
    self.constraints[key] = constraints

  @staticmethod
  def register_flags(config_class):
    register_flags(config_class)

  @check_flag_name
  def register(self, flag_name, *val):
    """Flag value can not be a tuple or a list"""
    assert len(val) > 0
    if len(val) == 1 and isinstance(val[0], (tuple, list)): val = val[0]

    if isinstance(val, (list, tuple)) and len(val) > 1:
      self.hyper_parameters[flag_name] = val
    else:
      if isinstance(val, (list, tuple)): val = val[0]
      self.common_parameters[flag_name] = val

  def run(self, times=1, save=False, mark=''):
    # Set the corresponding flags if save
    if save:
      self.common_parameters['save_model'] = True
    # Begin iteration
    counter = 0
    for _ in range(times):
      counter += 1
      if save: self.common_parameters['suffix'] = '_{}{}'.format(mark, counter)
      history = []
      for hyper_dict in self._hyper_parameter_dicts():
        params = self._get_all_configs(hyper_dict)
        self._apply_constraints(params)

        params_list = self._get_config_strings(params)
        params_string = ' '.join(params_list)
        if params_string in history: continue
        history.append(params_string)

        call(['python', self.module_name] + params_list)
        # call(self.command_head + params_list)
        print()

  # endregion : Public Methods

  # region : Private Methods

  def _apply_constraints(self, configs):
    assert isinstance(configs, dict)

    def _satisfy(condition):
      assert isinstance(condition, tuple) and len(condition) > 0
      for key, values in condition:
        if not isinstance(values, (tuple, list)): values = values,
        if key not in configs.keys(): return False
        elif configs[key] not in values: return False
      return True

    def _set_flag(flag_name, value):
      if flag_name not in flag_names:
        raise KeyError(
          '!! Failed to set `{}`  since it has not been registered'.format(
            flag_name))
      configs[flag_name] = value

    for condition, constraint in self.constraints.items():
      assert isinstance(constraint, dict)
      if _satisfy(condition):
        for key, value in constraint.items(): _set_flag(key, value)


  @staticmethod
  def _get_config_string(flag_name, val):
    return '--{}={}'.format(flag_name, val)

  @staticmethod
  def _get_config_strings(config_dict):
    assert isinstance(config_dict, dict)
    return [Helper._get_config_string(key, val) for key, val in
            config_dict.items()]

  def _get_all_configs(self, hyper_dict):
    assert isinstance(hyper_dict, OrderedDict)
    all_configs = OrderedDict()
    all_configs.update(self.common_parameters)
    all_configs.update(hyper_dict)
    return all_configs

  def _check_module(self):
    if not os.path.exists(self.module_name):
      raise AssertionError(
        '!! module {} does not exist'.format(self.module_name))

  def _hyper_parameter_dicts(self, keys=None):
    """Provide a generator of hyper-parameters for running"""
    if keys is None: keys = list(self.hyper_parameters.keys())
    if len(keys) == 0: yield {}
    else:
      for val in self.hyper_parameters[keys[0]]:
        configs = OrderedDict()
        configs[keys[0]] = val
        # cfg_str = self._get_config_string(keys[0], val)
        for cfg_dict in self._hyper_parameter_dicts(keys[1:]):
          configs.update(cfg_dict)
          yield configs
        # for cfg_list in self._hyper_parameter_lists(keys[1:]):
        #   yield [cfg_str] + cfg_list

  # endregion : Private Methods

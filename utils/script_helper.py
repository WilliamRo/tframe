from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import call

from tframe.configs.flag import Flag
from tframe.trainers import SmartTrainerHub


flags = [attr for attr in
         [getattr(SmartTrainerHub, key) for key in dir(SmartTrainerHub)]
         if isinstance(attr, Flag)]
flag_names = [f.name for f in flags]

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

    self.common_parameters = {}
    self.hyper_parameters = {}

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

  @check_flag_name
  def register(self, flag_name, val):
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
      for hyper in self._hyper_parameter_lists():
        call(self.command_head + hyper)
        print()

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _get_config_string(flag_name, val):
    return '--{}={}'.format(flag_name, val)

  def _check_module(self):
    if not os.path.exists(self.module_name):
      raise AssertionError(
        '!! module {} does not exist'.format(self.module_name))

  def _hyper_parameter_lists(self, keys=None):
    """Provide a generator of hyper-parameters for running"""
    if keys is None: keys = list(self.hyper_parameters.keys())
    if len(keys) == 0: yield []
    else:
      for val in self.hyper_parameters[keys[0]]:
        cfg_str = self._get_config_string(keys[0], val)
        for cfg_list in self._hyper_parameter_lists(keys[1:]):
          yield [cfg_str] + cfg_list

  # endregion : Private Methods

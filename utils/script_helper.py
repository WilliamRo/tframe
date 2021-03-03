from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import re
from subprocess import call
from collections import OrderedDict

from tframe import checker
from tframe import console
from tframe.utils.note import Note
from tframe.utils.local import re_find_single
from tframe.utils.misc import date_string
from tframe.utils.file_tools.imp import import_from_path
from tframe.configs.flag import Flag
from tframe.trainers import SmartTrainerHub

from tframe.alchemy.pot import Pot

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
    if flag_name in obj.sys_keys: return
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

  BAYESIAN = 'BAYESIAN'
  GRID_SEARCH = 'GRID_SEARCH'

  def __init__(self, module_name=None):
    self.module_name = module_name
    self._check_module()

    self.pot = Pot()

    self.common_parameters = OrderedDict()
    self.hyper_parameters = OrderedDict()
    self.constraints = OrderedDict()

    self._python_cmd = 'python' if os.name == 'nt' else 'python3'

    # System argv info. 'sys_keys' will be filled by _register_sys_argv.
    # Any config registered by Helper.register method with 1st arg in
    #  this list will be ignored. That is, system args have the highest priority
    self.sys_keys = []
    self._sys_runs = None           # (1)
    self._add_script_suffix = None  # (2)
    self._scroll = None             # (3)
    self._register_sys_argv()

  # region : Properties

  @property
  def hyper_parameter_keys(self):
    return self.pot.hyper_parameter_keys

  @property
  def command_head(self):
    return ['python', self.module_name] + [
      self._get_config_string(k, v) for k, v in self.common_parameters.items()]

  @property
  def param_keys(self):
    # Add keys from hyper-parameters
    keys = self.hyper_parameter_keys
    # Add keys from common-parameters
    keys += list(self.common_parameters.keys())
    return keys

  @property
  def default_summ_name(self):
    script_name = re_find_single(r's\d+_\w+(?=.py)')
    return '{}_{}'.format(date_string(), script_name)

  # endregion : Properties

  # region : Public Methods

  def constrain(self, conditions, constraints):
    # Make sure hyper-parameter keys have been registered.
    for key in list(conditions.keys()) + list(constraints.keys()):
      if key not in flag_names: raise KeyError(
          '!! Failed to set `{}`  since it has not been registered'.format(key))
    self.pot.constrain(conditions, constraints)

  @staticmethod
  def register_flags(config_class):
    register_flags(config_class)

  @check_flag_name
  def register(self, flag_name, *val):
    """Flag value can not be a tuple or a list"""
    if flag_name in self.sys_keys: return
    assert len(val) > 0
    if len(val) == 1 and isinstance(val[0], (tuple, list)): val = val[0]

    if isinstance(val, (list, tuple)) and len(val) > 1:
      self.pot.register_category(flag_name, val)
      # self.hyper__parameters[flag_name] = val   # TODO: deprecated
    else:
      if isinstance(val, (list, tuple)): val = val[0]
      self.common_parameters[flag_name] = val
      self._show_flag_if_necessary(flag_name, val)

  def set_python_cmd_suffix(self, suffix='3'):
    self._python_cmd = 'python{}'.format(suffix)

  def run(self, times=1, save=False, mark='', rehearsal=False, method='grid'):
    """Run script using the given 'method'. This method is compatible with
       old version of tframe script_helper, and should be deprecated in the
       future.
    """
    # :: Check options passed by system arguments
    # Check sys_runs
    if self._sys_runs is not None:
      times = checker.check_positive_integer(self._sys_runs)
      console.show_status('Run # set to {}'.format(times))
    # Set the corresponding flags if save. Here 'save' option is a short-cut
    # of doing 's.register('save_model', True)'
    if save: self.common_parameters['save_model'] = True
    # Set scroll for pot
    # method = method if self._scroll is None else self._scroll
    # > currently freeze method to 'grid' which will make 'run' method equal to
    #   'grid_search'.
    assert method == 'grid'

    # Show section
    console.section('Script Information')
    # Hyper-parameter info will be showed when scroll is set
    self.pot.set_scroll(method, times=times)
    # Show parameters
    self._show_parameters()

    # Begin iteration
    for i, config_dict in enumerate(self.pot.scroll.combinations()):
      # Handle script suffix option
      if self._add_script_suffix is not None:
        save = self._add_script_suffix
      if save:
        self.common_parameters['script_suffix'] = '_{}{}'.format(mark, i + 1)
      # Show hyper-parameters
      console.show_info('Hyper-parameters:')
      for k, v in config_dict.items():
        console.supplement('{}: {}'.format(k, v), level=2)
      # Run process if not rehearsal
      if rehearsal: continue
      console.split()
      call([self._python_cmd, self.module_name] + self._get_config_strings(
        config_dict))
      print()

  # endregion : Public Methods

  # region : Private Methods

  @staticmethod
  def _show_flag_if_necessary(flag_name, value):
    if flag_name == 'gpu_id':
      console.show_status('GPU ID set to {}'.format(value))
    if flag_name == 'gather_summ_name':
      console.show_status('Notes will be gathered to `{}`'.format(value))

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
    """If module name is not provided, try to find one according to the
       recommended project organization"""
    if self.module_name is None:
      self.module_name = '../t{}.py'.format(
        re_find_single(r'(?<=s)\d+_\w+(?=.py)'))
      console.show_status('Module set to `{}`'.format(self.module_name))

    if not os.path.exists(self.module_name):
      raise AssertionError(
        '!! module {} does not exist'.format(self.module_name))

  def _show_parameters(self):
    def _show_config(name, od):
      assert isinstance(od, OrderedDict)
      if len(od) == 0: return
      console.show_info(name)
      for k, v in od.items():
        console.supplement('{}: {}'.format(k, v), level=2)

    _show_config('Common Settings', self.common_parameters)
    # Hyper parameters and constraints will be showed by Scroll
    # _show_config('Constraints', self.constraints)

  def _register_sys_argv(self):
    """When script 'sX_YYYY.py' is launched using command line tools, system
       arguments other than tframe flags are allowed. These arguments, like
       tframe flags arguments passed via command line, also have the highest
       priority that will overwrite the corresponding arguments (if any) defined
       in related python modules.
    """
    # Check each system arguments
    for s in sys.argv[1:]:
      assert isinstance(s, str)
      # Check format (pattern: --flag_name=value)
      r = re.fullmatch(r'--([\w_]+)=([-\w./,+:;]+)', s)
      if r is None: raise AssertionError(
        'Can not parse argument `{}`'.format(s))
      # Parse key and value
      k, v = r.groups()
      assert isinstance(v, str)
      val_list = re.split(r'[,/]', v)
      # (1) Number of runs (take effect when method is grid search)
      if k in ('run', 'runs'):
        assert len(val_list) == 1
        self._sys_runs = checker.check_positive_integer(int(val_list[0]))
        continue
      # (2) Whether to save model. If set to true, script_suffix will be set
      #     as common parameter and run number will be automatically appended
      #     to model mark to ensure model created by this script_helper
      #     do not have the same mark
      if k in ('save', 'brand'):
        assert len(val_list) == 1
        option = val_list[0]
        assert option.lower() in ('true', 'false')
        self._add_script_suffix = option.lower() == 'true'
        continue
      # (3) Running method, can be grid, goose, bayesian, etc.
      if k in ('method', 'scroll'):
        assert len(val_list) == 1
        self._scroll = val_list[0]
        continue
      # Register key in common way
      self.register(k, *val_list)
      self.sys_keys.append(k)

  # endregion : Private Methods

  # region: Search Engine

  def _get_summary(self):
    """This method knows the location of summary files."""
    # Find summary file name
    key = 'gather_summ_name'
    assert key in self.common_parameters
    summ_file_name = self.common_parameters[key]
    # Get root_path
    task = import_from_path(self.module_name)
    task.update_job_dir(task.id, task.model_name)
    root_path = task.core.th.job_dir
    # Get summary path
    summ_path = os.path.join(root_path, summ_file_name)
    # Load notes if exists
    if os.path.exists(summ_path): return Note.load(summ_path)
    else: return []


  def search(self, criterion, greater_is_better, max_search_time=1000,
             expectation=None, method='goose', **kwargs):
    """Search hyper-parameters

    :param criterion: criterion key for HP search. Should be found in each
                      note of summary list
    :param greater_is_better: whether higher criterion is preferred
    :param kwargs: other arguments
    :param max_search_time: maximum search time
    :param expectation: if given, searching will be stopped if expectation has
                        been met
    :param method: search method
    :return: best criterion
    """
    # Show status
    console.section(
      'Searching hyper-parameters using {} engine'.format(method))

    # Set

    # TODO: Scaffold =======================================================
    notes = self._get_summary()

    console.show_status('Done.')


  # endregion: Search Engine

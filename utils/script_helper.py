from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys

# Register paths here
p = sys.path[0]
for _ in range(2):
  p = os.path.dirname(p)
  if not os.path.isdir(p): break
  sys.path.append(p)

import re
import time
from subprocess import run
from collections import OrderedDict

from tframe import console
from tframe.utils.note import Note
from tframe.utils.local import re_find_single
from tframe.utils.misc import date_string
from tframe.utils.file_tools.imp_tools import import_from_path
from tframe.utils.string_tools import get_time_string
from tframe.utils.file_tools.io_utils import safe_open
from tframe.utils.organizer.task_tools import update_job_dir
from tframe.configs.flag import Flag
from tframe.trainers import SmartTrainerHub

from tframe.alchemy.pot import Pot
from tframe.alchemy.scrolls import get_argument_keys

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
    if flag_name in obj.sys_flags: return
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

  class CONFIG_KEYS(object):
    add_script_suffix = 'add_script_suffix'
    auto_set_hp_properties = 'auto_set_hp_properties'
    strategy = 'strategy'
    criterion = 'criterion'
    greater_is_better = 'greater_is_better'
    python_version = 'python_version'
    python_cmd = 'python_cmd'

  def __init__(self, module_name=None):
    self.module_name = module_name
    self._check_module()

    self.pot = Pot(self._get_summary)

    self.common_parameters = OrderedDict()
    self.hyper_parameters = OrderedDict()
    self.constraints = OrderedDict()

    self._python_cmd = 'python' if os.name == 'nt' else 'python3'
    self._root_path = None

    # System argv info. 'sys_keys' will be filled by _register_sys_argv.
    # Any config registered by Helper.register method with 1st arg in
    #  this list will be ignored. That is, system args have the highest priority
    # USE SYS_CONFIGS WITH DATA TYPE CONVERSION!
    self.sys_flags = []
    self.config_dict = OrderedDict()
    self._init_config_dict()
    self._register_sys_argv()

  # region : Properties

  @property
  def criterion(self):
    return self.config_dict[self.CONFIG_KEYS.criterion]

  @property
  def configs(self):
    od = OrderedDict()
    for k, v in self.config_dict.items():
      if v is not None: od[k] = v
    return od

  @property
  def hyper_parameter_keys(self):
    return self.pot.hyper_parameter_keys

  @property
  def command_head(self):
    return ['python', self.module_name] + [
      self._get_hp_string(k, v) for k, v in self.common_parameters.items()]

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

  @property
  def summ_file_name(self):
    key = 'gather_summ_name'
    assert key in self.common_parameters
    return self.common_parameters[key]

  @property
  def shadow_th(self):
    task = import_from_path(self.module_name)
    return task.core.th

  @property
  def root_path(self):
    if self._root_path is not None: return self._root_path
    task = import_from_path(self.module_name)
    update_job_dir(task.id, task.model_name)
    self._root_path = task.core.th.job_dir
    return self.root_path

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
  def register(self, flag_name, *val, hp_type=None, scale=None):
    """Flag value can not be a tuple or a list"""
    assert len(val) > 0
    if len(val) == 1 and isinstance(val[0], (tuple, list)): val = val[0]

    if isinstance(val, (list, tuple)) and len(val) > 1:
      if flag_name in self.sys_flags:
        self.pot.set_hp_properties(flag_name, hp_type, scale)
      else: self.pot.register_category(flag_name, val, hp_type, scale)
    else:
      if flag_name in self.sys_flags: return
      if isinstance(val, (list, tuple)): val = val[0]
      self.common_parameters[flag_name] = val
      self._show_flag_if_necessary(flag_name, val)

  def set_hp_property(self, name, hp_type, scale=None):
    assert name in self.pot.hyper_parameter_keys
    self.pot.set_hp_properties(name, hp_type, scale)

  def configure(self, **kwargs):
    for k, v in kwargs.items():
      assert not isinstance(v, (tuple, list, set))
      if k not in self.config_dict:
        raise KeyError('!! Unknown system config `{}`'.format(k))
      # Set only when the corresponding config has not been set by
      #   system arguments
      if self.config_dict[k] is None: self.config_dict[k] = v

  def set_python_cmd_suffix(self, suffix='3'):
    self._python_cmd = 'python{}'.format(suffix)

  def run(self, strategy='grid', rehearsal=False, **kwargs):
    """Run script using the given 'strategy'. This method is compatible with
       old version of tframe script_helper, and should be deprecated in the
       future. """
    # Show section
    console.section('Script Information')
    # Show pot configs
    self._show_dict('Pot Configurations', self.configs)
    # Hyper-parameter info will be showed when scroll is set
    self.configure(**kwargs)
    # Do some auto configuring, e.g., set greater_is_better based on the given
    #   criterion
    self._auto_config()
    self.pot.set_scroll(self.configs.get('strategy', strategy), **self.configs)
    # Show common parameters
    self._show_dict('Common Settings', self.common_parameters)

    # Begin iteration
    for i, hyper_params in enumerate(self.pot.scroll.combinations()):
      # Show hyper-parameters
      console.show_info('Hyper-parameters:')
      for k, v in hyper_params.items():
        console.supplement('{}: {}'.format(k, v), level=2)
      # Run process if not rehearsal
      if rehearsal: continue
      console.split()
      # Export log if necessary
      if self.pot.logging_is_needed: self._export_log()
      # Run
      self._run_process(hyper_params, i)

  # endregion : Public Methods

  # region : Private Methods

  def _init_config_dict(self):
    """Config keys include that specified in self.CONFIG_KEYS and
       the primary arguments in the constructor of sub-classes of Scroll"""
    assert isinstance(self.config_dict, OrderedDict)
    # Add all keys specified in CONFIG_KEYS
    key_list = [k for k in self.CONFIG_KEYS.__dict__ if k[:2] != '__']
    # Get add keys from scroll classes
    key_list += get_argument_keys()
    # Initialize these configs as None
    # key_list may contain duplicated keys, which is OK
    for key in key_list: self.config_dict[key] = None

  def _auto_config(self):
    if self.configs.get(self.CONFIG_KEYS.strategy, 'grid') == 'grid': return
    # Try to automatically set greater_is_better
    if self.CONFIG_KEYS.greater_is_better not in self.configs:
      # criterion = self.config_dict[self.CONFIG_KEYS.criterion]
      criterion = self.criterion
      if isinstance(criterion, str):
        criterion = criterion.lower()
        if any([s in criterion for s in ('accuracy', 'f1', 'improvement')]):
            self.config_dict[self.CONFIG_KEYS.greater_is_better] = True
        elif any([s in criterion for s in (
            'loss', 'cross_entropy', 'bpc', 'perplexity')]):
            self.config_dict[self.CONFIG_KEYS.greater_is_better] = False

    # Try to set hyper-parameters' properties
    if self.configs.get(self.CONFIG_KEYS.auto_set_hp_properties, True):
      self.auto_set_hp_properties()

  def _run_process(self, hyper_params, index):
    assert isinstance(hyper_params, dict)
    # Handle script suffix option
    if self.configs.get('add_script_suffix', False):
      self.common_parameters['script_suffix'] = '_{}'.format(index + 1)
    # Run
    configs = self._get_all_configs(hyper_params)
    cmd = self._python_cmd.split(' ') + [self.module_name]
    cmd += self._get_hp_strings(configs)
    console.show_status(f'Executing `{cmd}` ...', symbol='[s-file]')
    run(cmd)
    print()

  @staticmethod
  def _show_flag_if_necessary(flag_name, value):
    if flag_name == 'gpu_id':
      console.show_status('GPU ID set to {}'.format(value))
    if flag_name == 'gather_summ_name':
      console.show_status('Notes will be gathered to `{}`'.format(value))

  @staticmethod
  def _get_hp_string(flag_name, val):
    return '--{}={}'.format(flag_name, val)

  @staticmethod
  def _get_hp_strings(config_dict):
    assert isinstance(config_dict, dict)
    return [Helper._get_hp_string(key, val) for key, val in
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
      task_path = os.path.dirname(sys.path[0])
      self.module_name = '{}/t{}.py'.format(
        task_path, re_find_single(r'(?<=s)\d+_\w+(?=.py)'))
      console.show_status('Module set to `{}`'.format(self.module_name))

    if not os.path.exists(self.module_name):
      raise AssertionError(
        '!! module {} does not exist'.format(self.module_name))

  @staticmethod
  def _show_dict(name, od):
    assert isinstance(od, OrderedDict)
    if len(od) == 0: return
    console.show_info(name)
    for k, v in od.items():
      console.supplement('{}: {}'.format(k, v), level=2)

  def _register_sys_argv(self):
    """When script 'sX_YYYY.py' is launched using command line tools, system
       arguments other than tframe flags are allowed. These arguments, like
       tframe flags arguments passed via command line, also have the highest
       priority that will overwrite the corresponding arguments (if any) defined
       in related python modules.

       TODO: this method should be refactored
    """
    # Check each system arguments
    for s in sys.argv[1:]:
      assert isinstance(s, str)
      # Check format (pattern: --flag_name=value)
      r = re.fullmatch(r'--([\w_]+)=([-\w./,+:; ]+)', s)
      if r is None: raise AssertionError(
        'Can not parse argument `{}`'.format(s))
      # Parse key and value
      k, v = r.groups()
      assert isinstance(v, str)

      # [Workaround] Patch for python_cmd
      if k == self.CONFIG_KEYS.python_cmd:
        self._python_cmd = v
        sys.argv.remove(s)
        continue

      val_list = re.split(r'[,/]', v)
      # Check system configurations
      if k in self.config_dict:
        assert len(val_list) == 1
        val = val_list[0]
        if k == self.CONFIG_KEYS.python_version:
          self._python_cmd = 'python{}'.format(val)
        else: self.config_dict[k] = val
        sys.argv.remove(s)
        continue

      # Register key in common way
      self.register(k, *val_list)
      self.sys_flags.append(k)

  # endregion : Private Methods

  # region: Search Engine Related

  def _get_summary(self):
    """This method knows the location of summary files."""
    # Get summary path
    summ_path = os.path.join(self.root_path, self.summ_file_name)
    # Load notes if exists
    if os.path.exists(summ_path):
      return [self._handle_hp_alias(n) for n in Note.load(summ_path)]
    else: return []

  @staticmethod
  def _handle_hp_alias(note):
    """TODO: this method is a compromise for HP alias, such as `lr`"""
    alias_dict = {'learning_rate': 'lr'}
    for name, alias in alias_dict.items():
      if name in note.configs:
        note.configs[alias] = note.configs.pop(name)
    return note

  def configure_engine(self, **kwargs):
    """This method is used for providing argument specifications in smart IDEs
       such as PyCharm"""
    kwargs.get('acq_kappa', None)
    kwargs.get('acq_n_points', None)
    kwargs.get('acq_n_restarts_optimizer', None)
    kwargs.get('acq_optimizer', None)
    kwargs.get('acq_xi', None)
    kwargs.get('acquisition', None)
    kwargs.get('add_script_suffix', None)
    kwargs.get('auto_set_hp_properties', None)
    kwargs.get('criterion', None)
    kwargs.get('expectation', None)
    kwargs.get('greater_is_better', None)
    kwargs.get('initial_point_generator', None)
    kwargs.get('n_initial_points', None)
    kwargs.get('prior', None)
    kwargs.get('strategy', None)
    kwargs.get('times', None)
    kwargs.get('python_version', None)
    self.configure(**kwargs)

  def _export_log(self):
    # Determine filename
    log_path = os.path.join(
      self.root_path, '{}_log.txt'.format(self.pot.scroll.name))
    # Get log from scroll
    engine_logs = self.pot.scroll.get_log()
    # Create if not exist
    with safe_open(log_path, 'a'): pass
    # Write log at the beginning
    with safe_open(log_path, 'r+') as f:
      content = f.readlines()
      f.seek(0)
      f.truncate()
      f.write('{} summ: {}, criterion: {}, scroll: {} \n'.format(
        get_time_string(), self.summ_file_name, self.criterion,
        self.pot.scroll.details))
      for line in engine_logs: f.write('  {}\n'.format(line))
      f.write('-' * 79 + '\n')
      f.writelines(content)

  def auto_set_hp_properties(self):
    """Set the properties of hyper-parameters automatically"""
    from tframe.alchemy.hyper_param import HyperParameter
    from tframe.configs.flag import Flag

    HubClass = type(self.shadow_th)
    for hp in self.pot.hyper_params:
      assert isinstance(hp, HyperParameter)
      # Find the corresponding flag in th
      flag = getattr(HubClass, hp.name)
      assert isinstance(flag, Flag)
      if hp.hp_type is None and flag.hp_type is not None:
        hp.hp_type = flag.hp_type
        console.show_status(
          '{}.hp_type set to {}'.format(hp.name, hp.hp_type), '[AutoSet]')
      if hp.scale is None and flag.hp_scale is not None:
        hp.scale = flag.hp_scale
        console.show_status(
          '{}.scale set to {}'.format(hp.name, hp.scale), '[AutoSet]')

  # endregion: Search Engine Related

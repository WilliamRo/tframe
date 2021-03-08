import numpy as np

from collections import OrderedDict
from tframe import console
from tframe.utils.note import Note

from .scrolls import get_scroll, Scroll
from .hyper_param import HyperParameter
from .hyper_param import BooleanHP, CategoricalHP, FloatHP, IntegerHP


class Pot(object):
  """This class serves as a good bridge between tframe facilities and pure
     search method."""

  class KEYS(object):
    CRITERION = 'criterion'

  def __init__(self, summary_fetcher):
    self.scroll = None
    self.hyper_params = []
    self.constraints = OrderedDict()
    self.criterion = None
    # Summary related variables
    assert callable(summary_fetcher)
    self.summary_fetcher = summary_fetcher

  @property
  def hyper_parameter_keys(self):
    return [hp.name for hp in self.hyper_params]

  @property
  def logging_is_needed(self):
    assert isinstance(self.scroll, Scroll)
    return self.scroll.logging_is_needed

  # region: Methods for HP registration and constraints setting
  # These method will be called by ScriptHelper before running

  def register_category(self, name, values, hp_type=None, scale=None):
    if all([len(values) == 2,
            set(values) in ({True, False}, {'true', 'false'})]):
      self.hyper_params.append(BooleanHP(name))
    else:
      self.hyper_params.append(CategoricalHP(name, values, hp_type, scale))

  def set_hp_properties(self, name, hp_type=None, scale=None):
    """This method is used to preserve the configs specified in sXX_YYY.py.
    If HP properties is passed through sys args, properties set in sXX_YYY.py
    will be ignored. If sys args do not specify HP properties, sXX_YYY.py
    property specification will take effect. """
    hp_list = [hp for hp in self.hyper_params if hp.name == name]
    assert len(hp_list) == 1
    hp = hp_list[0]
    assert isinstance(hp, HyperParameter)
    if hp.hp_type is None: hp.hp_type = hp_type
    if hp.scale is None: hp.scale = scale

  def register_range(self, name, v_min, v_max, val_type=int, scale='uniform'):
    assert val_type in (int, float)
    HP = IntegerHP if val_type is int else FloatHP
    self.hyper_params.append(HP(name, v_min, v_max, scale))

  def constrain(self, conditions, constraints):
    """Constrain hyper-parameters combination.

    SYNTAX:
     p.constrain({'lr': 0.001, 'batch_size': 32}, {'dropout': 0.1})

     p.constrain({'lr': 0.001, 'batch_size': (16, 128)},
                 {'dropout': (0.1, 0.2), 'batch_norm': False})

    :param conditions: condition dictionary
    :param constraints: constraint dictionary
    """
    assert all([isinstance(c, dict) for c in (conditions, constraints)])
    key = tuple(sorted([(k, v) for k, v in conditions.items()]))
    self.constraints[key] = constraints

  # endregion: Methods for HP registration and constraints setting

  # region: Bridges Between Helper and Scroll

  def set_scroll(self, identifier, **kwargs):
    """This method is called by Helper.run method"""
    # scroll is allowed to be set only once
    if self.scroll is not None: raise AssertionError(
      '!! Scroll has already been set to {}'.format(self.scroll.name))
    # Get scroll class
    ScrollClass = get_scroll(identifier)
    # Initiate a scroll for this pot
    self.scroll = ScrollClass(
      self.hyper_params, self.constraints,
      observation_fetcher=self.get_observations, **kwargs)
    # Set criterion
    self.criterion = kwargs.get(self.KEYS.CRITERION, None)

  def get_observations(self):
    """Translate notes to list of dictionaries. If this method is called,
       hyper_params and criterion must be given. """
    # Check hyper_params and criterion
    if len(self.hyper_params) == 0:
      raise AssertionError('!! Hyper-Parameters has not been set.')
    # Check criterion
    if self.criterion is None:
      raise AssertionError(
        '!! Criterion for hyper-parameter searching has not been set.')
    # Fetch notes
    notes = self.summary_fetcher()
    if len(notes) == 0: return []
    # Peel of Note wrapper
    observations = []
    for note in notes:
      # Every note in the note list must contain the criterion
      if self.criterion not in note.criteria: raise AssertionError(
        '!! Every note must contain the criterion `{}`'.format(self.criterion))
      # This note will be ignored if it does not contain all the information
      #  in self.hyper_params or the config value is not within the range
      if not all([hp.name in note.configs and hp.within(note.configs[hp.name])
                  for hp in self.hyper_params]): continue
      # Gather observation
      od = OrderedDict()
      # self.scroll.hyper_params.values() may have been found themselves
      for hp in self.scroll.hyper_params.values():
        assert isinstance(hp, HyperParameter)
        od[hp] = note.configs[hp.name]
      # Organize the observation list as a list of tuples
      observations.append((od, note.criteria[self.criterion]))
    return observations

  # endregion: Bridges Between Helper and Scroll

  # region: Private Methods


  # endregion: Private Methods


import numpy as np

from collections import OrderedDict
from tframe import console

from .scrolls import get_scroll
from .hyper_param import BooleanHP, CategoricalHP, FloatHP, IntegerHP


class Pot(object):
  """This class serves as a good bridge between tframe facilities and pure
     search method."""

  def __init__(self):
    self.scroll = None
    self.hyper_params = []
    self.constraints = OrderedDict()


  @property
  def hyper_parameter_keys(self):
    return [hp.name for hp in self.hyper_params]


  def set_scroll(self, identifier, **kwargs):
    # scroll is allowed to be set only once
    if self.scroll is not None: raise AssertionError(
      '!! Scroll has already been set to {}'.format(self.scroll.name))
    # Get scroll class
    ScrollClass = get_scroll(identifier)
    console.show_status(
      'Pot with {} has been created.'.format(ScrollClass.name))
    # Initiate a scroll for this pot
    self.scroll = ScrollClass(self.hyper_params, self.constraints, **kwargs)


  # region: Methods for HP registration and constraints setting
  # These method will be called by ScriptHelper before running

  def register_category(self, name, values):
    if all([len(values) == 2,
            set(values) in ({True, False}, {'true', 'false'})]):
      self.hyper_params.append(BooleanHP(name))
    else: self.hyper_params.append(CategoricalHP(name, values))

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

  def update(self):
    pass








from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
from collections import OrderedDict


class Plugin(object):
  """Base class for plugins to tensor viewer"""
  def __init__(self, dict_modifier):
    assert callable(dict_modifier)
    self._dict_modifier = dict_modifier

  def modify_variable_dict(self, v_dict):
    assert isinstance(v_dict, OrderedDict)
    self._dict_modifier(v_dict)


class VariableWithView(object):

  def __init__(self, value_list, view):
    self._value_list = value_list
    assert callable(view)
    self._view = view

  def display(self, vv): self._view(vv, self._value_list)


def recursively_modify(method, v_dict, level=0, verbose=True):
  """This method recursively modifies v_dict with a provided 'method'.
     'method' accepts keys and values(list of numpy arrays) and returns
     modified values (which can be a tframe.VariableViewer).
     Sometimes method should contain logic to determine whether the input values
     should be modified.
  """
  # Sanity check
  assert callable(method) and isinstance(v_dict, dict)
  assert inspect.getfullargspec(method).args == ['key', 'value']
  if len(v_dict) == 0: return

  # If values in v_dict are dictionaries,  recursively modify each of them
  if isinstance(list(v_dict.values())[0], dict):
    for e_key, e_dict in v_dict.items():
      assert isinstance(e_dict, dict)
      if verbose: print('*> modifying dict {} ...'.format(e_key))
      recursively_modify(method, e_dict, level=level+1, verbose=verbose)
    return

  # At this point, values in v_dict must be lists of numpy arrays
  for key in v_dict.keys(): v_dict[key] = method(key, v_dict[key])

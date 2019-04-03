from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

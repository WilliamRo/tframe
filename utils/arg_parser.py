from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from collections import OrderedDict


class Parser(object):
  """USAGE
     p = Parser.parse('max_norm:3.0;axis=1')

     print(p.get_kwarg('axis', int))
  >> 1

  """

  def __init__(self, *args, splitter='=', ignore_in_suffix=(), **kwargs):
    self.name = None
    self.arg_list = list(args)
    self.arg_dict = OrderedDict(kwargs)
    if not isinstance(ignore_in_suffix, (tuple, list)):
      ignore_in_suffix = (ignore_in_suffix,)
    self.ignore_in_suffix = ignore_in_suffix
    # Check splitter
    assert isinstance(splitter, str) and len(splitter) == 1
    self.splitter = splitter


  def __getitem__(self, key):
    if key in self.arg_dict: return self.arg_dict[key]
    raise KeyError('!! Key `{}` not found'.format(key))


  @property
  def filename_suffix(self):
    name = self.name if self.name else 'noname'
    suffix = ''
    if len(self.arg_list) > 0: suffix += ','.join(self.arg_list)
    if len(self.arg_dict) > 0: suffix += ','.join(
      ['{}={}'.format(k, v) for k, v in self.arg_dict.items()
       if k not in self.ignore_in_suffix])
    if suffix is not '': suffix = '({})'.format(suffix)
    return name + suffix


  # Deprecated
  def get_arg(self, dtype=str, default=None):
    """Used in modules like activation.py for parsing `lrelu:0.15`"""
    if len(self.arg_list) == 0 and default is not None: return default
    assert len(self.arg_list) == 1
    arg = self.arg_list[0]
    return dtype(arg)


  def get_kwarg(self, key, dtype=str, default=None):
    assert isinstance(key, str)
    if key not in self.arg_dict:
      if default is not None: return default
      else: raise KeyError('Key `{}` does not exist.'.format(key))
    # Key is in self.arg_dict
    val_str = self.arg_dict[key]
    assert isinstance(val_str, str)
    if dtype is bool:
      if val_str.lower() not in ('true', 'false'):
        raise ValueError('Illegal bool string `{}`'.format(val_str))
      return val_str.lower() == 'true'
    return dtype(self.arg_dict[key])


  def parse_arg_string(self, arg_string):
    assert isinstance(arg_string, str) and len(arg_string) > 0
    name_and_args = arg_string.split(':')
    if len(name_and_args) > 2: raise AssertionError(
      '!! Can not parse `{}`, too many `:` found.'.format(arg_string))
    # Pop key
    self.name = name_and_args[0]
    # Parse args
    if len(name_and_args) == 1: return
    self._parse_arg_list(re.split('[,;]', name_and_args[1]))


  def _parse_arg_list(self, args):
    assert isinstance(args, (tuple, list)) and len(args) > 0
    # val = None
    for arg in args:
      kv_list = arg.split(self.splitter)
      assert len(kv_list) > 0
      if len(kv_list) > 2:
        raise AssertionError('!! Can not resolve `{}`'.format(arg))
      if len(kv_list) == 1:
        self.arg_list.append(kv_list[0])
        # if val is not None:
        #   raise ValueError('!! There are too many non-kw args')
        # val = kv_list[0]
        # continue
      else: self.arg_dict[kv_list[0]] = kv_list[1]
    # if val is not None: self.arg_list = [val]


  @staticmethod
  def parse(arg_string, *args, splitter='=', ignore_in_suffix=(), **kwargs):
    p = Parser(
      *args, splitter=splitter, ignore_in_suffix=ignore_in_suffix, **kwargs)
    p.parse_arg_string(arg_string)
    return p


if __name__ == '__main__':
  p = Parser.parse('bayers:low=1;high=10.1', splitter='=')
  print(p.name)
  print(p.get_kwarg('low', float))
  print(p.get_kwarg('high', float))
  print(p.get_kwarg('type', str, default='float'))

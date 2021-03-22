from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict


class Nomear(object):
  """This class is designed to provide a `pocket` dictionary for instances
  of classes inherited from it. If you are curious about its name, append
  `od` and reverse. One of the cool things about this class is that users
  do not need to call its constructor/initializer.

  This class follows the coding philosophy that some fields of a certain
  class do not need to be explicitly appear in constructor. Sometimes put
  them into a pocket is more comfortable.
  """

  _4D_POCKET = '_4D_POCKET'

  _register = OrderedDict()


  @property
  def _pocket(self) -> OrderedDict:
    # If self is not registered, register
    if self not in self._register: self._register[self] = OrderedDict()
    return self._register[self]


  def get_from_pocket(self, key: str, default=None, initializer=None):
    if key not in self._pocket:
      if callable(initializer): self._pocket[key] = initializer()
      else: return default
    return self._pocket[key]


  def put_into_pocket(self, key: str, thing, exclusive=True):
    if key in self._pocket and exclusive:
      raise KeyError("`{}` already exists in {}'s pocket.".format(key, self))
    self._pocket[key] = thing


  def replace_stuff(self, key: str, val):
    assert key in self._pocket
    self._pocket[key] = val


  def __getitem__(self, item):
    return self.get_from_pocket(item)


  def __setitem__(self, key, value):
    self.put_into_pocket(key, value)

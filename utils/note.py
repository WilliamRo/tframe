from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Note(object):
  def __init__(self):
    self._lines = []

  # region : Properties

  @property
  def content(self):
    return '\n'.join(self._lines)

  # endregion : Properties

  # region : Public Methods

  def write_line(self, line):
    assert isinstance(line, str)
    self._lines.append(line)

  # endregion : Public Methods

  # region : Private Methods
  # endregion : Private Methods

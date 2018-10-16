from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import checker


def table_row(cols, widths):
  """
  Generate a table row
  :param cols: a list or a tuple of strings
  :param widths: a list or a tuple of positive integer
  :return: a string of length sum(widths)
  """
  # Sanity check
  assert isinstance(cols, (tuple, list)) and isinstance(widths, (tuple, list))
  assert len(cols) == len(widths)
  checker.check_type(widths, int)

  # Add column to row one by one
  row = ''
  for c, w in zip(cols, widths):
    assert isinstance(c, str)
    # If len(c) > w, truncate string c
    if len(c) > w:
      col = c[:w]
    else:
      col = c + ' ' * (w - len(c))
    # Add col to row
    row += col

  return row




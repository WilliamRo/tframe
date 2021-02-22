from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class DisplayConfig(object):
  """Configurations for displaying"""

  structure_detail_widths = Flag.list(
    [40, 22, 15], 'Widths of columns displayed in model structure detail')


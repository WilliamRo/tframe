from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from tframe import config
from tframe import console
from tframe import FLAGS
from tframe import pedia
from tframe import with_graph


class Frame(object):
  """Basic frame containing the functions of
     (1) saving and loading the inherited model
     (2) manipulating the path of logs, checkpoint files and snapshots
     (3) interfaces of building and training"""
  model_name = 'default'

  def __init__(self, mark=None):
    # If mark is not specified or is specified in FLAGS, use FLAGS.mark
    self.mark = (FLAGS.mark if mark is None or FLAGS.mark != pedia.default
                 else mark)

    # Placeholders for frame fields
    self._graph = None
    self._session = None
    self._summary_writer = None
    self._saver = None

  # region : Properties


  # endregion : Properties

  # region : Interfaces
  # endregion : Interfaces

  # region : Public Methods


  # endregion : Public Methods

  # region : Private Methods


  # endregion : Private Methods

  """For some reasons, do not remove this line"""

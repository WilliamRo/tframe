from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from tframe import checker
from tframe import console


class ProgressBar(object):

  def __init__(self, total):
    self._total = checker.check_positive_integer(total)
    self._start_time = time.time()

  def tic(self):
    self._start_time = time.time()

  def show(self, index):
    console.print_progress(index, self._total, start_time=self._start_time)

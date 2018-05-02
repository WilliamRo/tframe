from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class TrainScheme(object):
  def __init__(self):
    # Attributes
    self._trials = []

  def enqueue(self, trial):
    assert isinstance(trial, Trial)
    self._trials.append(trial)

  def dequeue(self):
    if len(self._trials) == 0: return None
    trial = self._trials[0]
    assert isinstance(trial, Trial)
    self._trials.remove(trial)
    return trial


class Trial(object):
  def __init__(self, init_function):
    # Private attributes
    self._train_steps = []
    self._learning_rates = []

    assert callable(init_function)
    self._init_function = None

  def initialize(self, model):
    self._init_function(model)


from collections import OrderedDict

import numpy as np

from tframe import console

from tframe.alchemy.scrolls.scroll_base import Scroll
from tframe.alchemy.hyper_param import CategoricalHP, FloatHP, HyperParameter

from .priors import get_prior


class Bayesian(Scroll):
  """TODO: this class is unfinished yet"""

  name = 'Bayesian'
  valid_HP_types = (CategoricalHP, FloatHP)
  enable_hp_types = True

  def __init__(self, hyper_params, constraints, observation_fetcher,
               prior='gp', acquisition='ei', times=None, expectation=None,
               **kwargs):
    # Call parent's constructor
    super(Bayesian, self).__init__(
      hyper_params, constraints, observation_fetcher=observation_fetcher,
      **kwargs)

    # Specific variables
    assert prior == 'gp' and acquisition == 'ei'  # fix for now
    self.prior = get_prior(prior)(**kwargs)
    self.acquisition = acquisition
    self.times = times
    self.expectation = expectation

    # Buffers
    self.previous_observations = None


  def combinations(self):
    run_id = 0
    while True:
      # Increase run_id
      run_id += 1
      # Get observations
      xs, ys = self._get_vector_observations()
      # Decide whether to terminate
      if any([self.times is not None and run_id > self.times,
              self.expectation is not None and any([
                self.is_better(y, self.expectation) for y in ys])]): break

      # Fit prior

      # Sample next hyper-parameter

      # Yield result
      yield None

  # region: Transformations

  def _get_vector_observations(self):
    observations = self.observation_fetcher()
    xs, ys = [], []
    for ob in observations:
      assert isinstance(ob, tuple) and len(ob) == 2
      hp, criterion = ob
      assert isinstance(hp, dict) and isinstance(criterion, float)
      xs.append(np.concatenate([k.to_vector_list(v) for k, v in hp.items()]))
      ys.append(criterion)
    return xs, ys

  def input_transform(self, x):
    return x


  # endregion: Transformations

  # region: Private Methods


  # endregion: Private Methods



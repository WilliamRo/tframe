import numpy as np
import time

from skopt import Optimizer
from skopt.space import Real, Integer, Categorical

from tframe import console
from tframe.alchemy.scrolls.scroll_base import Scroll
from tframe.alchemy.hyper_param import CategoricalHP, FloatHP
from tframe.alchemy.hyper_param import HyperParameter, IntegerHP


class Bayesian(Scroll):

  name = 'SK-BOGP'
  valid_HP_types = (CategoricalHP, FloatHP)

  enable_hp_types = True
  logging_is_needed = True

  def __init__(self, hyper_params, constraints, observation_fetcher,
               prior='gp', acquisition='gp_hedge', times=None, expectation=None,
               n_initial_points=5, initial_point_generator='random',
               acq_optimizer='lbfgs', **kwargs):
    """TODO list:
       - export logs
       - k(T(x_i), T(x_j))
    """
    # Call parent's constructor
    super(Bayesian, self).__init__(
      hyper_params, constraints, observation_fetcher=observation_fetcher,
      **kwargs)

    # Specific variables
    assert prior == 'gp'  # fix for now
    self.prior = prior
    self.acquisition = acquisition
    self.times = times
    self.expectation = expectation

    # skopt attributes
    self.dimensions = []
    self._fill_dimension_list()

    # Initialize a skopt optimizer
    self.n_initial_points = n_initial_points
    self.acq_optimizer = acq_optimizer
    self.optimizer = Optimizer(self.dimensions, prior, acq_func=acquisition,
                               n_initial_points=n_initial_points,
                               initial_point_generator=initial_point_generator,
                               acq_optimizer=acq_optimizer)

  @property
  def details(self):
    return '{} (init: {}, prior: {}, acq: {}, acq_opt: {})'.format(
      self.name, self.n_initial_points, self.prior, self.acquisition,
      self.acq_optimizer)

  # region: Private Methods

  def _fill_dimension_list(self):
    """Transform tframe hyper-parameters to skopt dimensions"""
    for hp in self.hyper_params.values():
      assert isinstance(hp, HyperParameter)
      # Transform HP according to type
      if isinstance(hp, FloatHP):
        DimClass = {IntegerHP: Integer, FloatHP: Real}[type(hp)]
        dimension = DimClass(hp.v_min, hp.v_max, prior=hp.scale, name=hp.name)
      elif isinstance(hp, CategoricalHP):
        dimension = Categorical(hp.choices, name=hp.name)
      else: raise TypeError('!! Unknown HP type `{}`'.format(type(hp)))
      # Append dimension to dimension list
      self.dimensions.append(dimension)

  # endregion: Private Methods

  def combinations(self):
    run_id = 0
    while True:
      # Increase run_id
      run_id += 1
      self.log('Run # {}'.format(run_id))
      # Get new observations
      new_X_y_list = self.get_new_x_y()
      xs, ys = [[xy[i] for xy in new_X_y_list] for i in range(2)]
      # Decide whether to terminate
      if self.times is not None and run_id > self.times:
        self.log('Terminate on max run time ({}) achieved.'.format(self.times))
        break
      if self.expectation is not None and any([
        self.is_better(y, self.expectation) for y in ys]):
        self.log('Terminate since expectation ({}) is satisfied.'.format(
          self.expectation))
        break

      # Tell and ask
      if len(xs) > 0:
        tic = time.time()
        # Observe
        self.optimizer.tell(xs, ys, fit=True)
        detail = ' | Observed {}: {}'.format(len(xs), ', '.join(
          ['({}) {:.3f}'.format(i + 1, y) for i, y in enumerate(ys)]))
        detail += ' | BEST: {:.3f}'.format(self.best_criterion)
        detail += ' | fit time: {:.2f} sec'.format(time.time() - tic)
        self.log_strings[-1] += detail
      next_x = self.optimizer.ask()
      next_config = self._value_list_to_config(next_x)
      self.log('Next config: {}'.format(next_config))
      # Convert next_x to config and return
      yield next_config


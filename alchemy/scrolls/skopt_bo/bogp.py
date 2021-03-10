import numpy as np
import time

from skopt import Optimizer
from skopt.space import Real, Integer, Categorical

from tframe import console
from tframe.alchemy.scrolls.scroll_base import Scroll
from tframe.alchemy.hyper_param import CategoricalHP, FloatHP
from tframe.alchemy.hyper_param import HyperParameter, IntegerHP

from .estimators import get_base_estimator


class Bayesian(Scroll):

  name = 'SK-BOGP'
  valid_HP_types = (CategoricalHP, FloatHP)

  enable_hp_types = True
  logging_is_needed = True

  def __init__(self, hyper_params, constraints, observation_fetcher,
               prior='gp', acquisition='gp_hedge', times=None, expectation=None,
               n_initial_points=5, initial_point_generator='random',
               acq_optimizer='auto', acq_n_points=10000, acq_xi=0.01,
               acq_n_restarts_optimizer=5, acq_kappa=1.96, **kwargs):
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
    self.acq_n_points = acq_n_points
    self.acq_xi = acq_xi
    self.acq_kappa = acq_kappa
    self.acq_n_restarts_optimizer = acq_n_restarts_optimizer

    acq_optimizer_kwargs = {
      "n_points": acq_n_points,
      "n_restarts_optimizer": acq_n_restarts_optimizer}
    acq_func_kwargs = {"xi": acq_xi, "kappa": acq_kappa}

    # Get estimator
    base_est = prior
    if isinstance(prior, str) and prior.lower() == 'gp':
      base_est = get_base_estimator(prior, self.dimensions)
    # Create optimizer
    self.optimizer = Optimizer(self.dimensions, base_est,
                               acq_func=self.acquisition,
                               n_initial_points=self.n_initial_points,
                               initial_point_generator=initial_point_generator,
                               acq_optimizer=self.acq_optimizer,
                               acq_func_kwargs=acq_func_kwargs,
                               acq_optimizer_kwargs=acq_optimizer_kwargs)

  @property
  def details(self):
    return '{} ({})'.format(self.name, ', '.join([
      '{}: {}'.format(k, v) for k, v in {
        'n_init': self.n_initial_points, 'prior': self.prior,
        'acq': self.acquisition, 'acq_opt': self.acq_optimizer,
        'acq_xi': self.acq_xi, 'acq_kappa': self.acq_kappa,
        'acq_n_points': self.acq_n_points}.items()]))

  # region: Private Methods

  def _check_acq_optimizer(self):
    """This method is not used for now"""
    if all([isinstance(d, Categorical) for d in self.dimensions]):
      self.acq_optimizer = 'sampling'

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
        # If greater_is_better, reverse the sign
        ys_to_tell = [-y for y in ys] if self.greater_is_better else ys
        # Observe
        self.optimizer.tell(xs, ys_to_tell, fit=True)
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


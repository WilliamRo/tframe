import numpy as np


def _get_gp_est(space, **kwargs):
  from skopt.utils import Space
  from skopt.utils import normalize_dimensions
  from skopt.utils import ConstantKernel, HammingKernel, Matern
  from skopt.learning import GaussianProcessRegressor

  # Set space
  space = Space(space)
  space = Space(normalize_dimensions(space.dimensions))
  n_dims = space.transformed_n_dims


  cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))
  # If all dimensions are categorical, use Hamming kernel
  if space.is_categorical:
    other_kernel = HammingKernel(length_scale=np.ones(n_dims))
  else:
    other_kernel = Matern(
      length_scale=np.ones(n_dims),
      length_scale_bounds=[(0.01, 100)] * n_dims, nu=2.5)

  base_estimator = GaussianProcessRegressor(
    kernel=cov_amplitude * other_kernel,
    normalize_y=True, noise="gaussian",
    n_restarts_optimizer=2)

  base_estimator.set_params(**kwargs)
  return base_estimator


def get_gp(dimensions):
  from skopt.utils import check_random_state

  rng = check_random_state(None)
  random_state = rng.randint(0, np.iinfo(np.int32).max)
  base_estimator = _get_gp_est(space=dimensions, random_state=random_state)
  return base_estimator

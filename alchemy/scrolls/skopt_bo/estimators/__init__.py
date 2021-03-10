from .gp import get_gp


def get_base_estimator(est_str, dimensions):
  assert isinstance(est_str, str) and est_str.lower() == 'gp'
  return get_gp(dimensions)
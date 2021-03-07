from .prior_base import PriorBase
from .gp import GaussianProcess


PRIORS = {
  'gp': GaussianProcess,
  'gaussian_process': GaussianProcess,
}


def get_prior(identifier):
  if isinstance(identifier, PriorBase): return identifier
  assert isinstance(identifier, str)
  s = identifier.lower().replace('-', '_')
  if not s in PRIORS:
    raise KeyError("!! Unknown prior key '{}'".format(s))
  return PRIORS[s]

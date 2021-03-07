from .scroll_base import Scroll
from .bo.bayesian import Bayesian
from .goose import Goose
from .grid_search import GridSearch


SCROLL_SHELF = {
  'bayes': Bayesian,
  'bayesian': Bayesian,
  'grid': GridSearch,
  'grid_search': GridSearch,
  # 'goose': Goose,
}


def get_scroll(s):
  if isinstance(s, Scroll): return s
  assert isinstance(s, str)
  s = s.lower().replace('-', '_')

  # Deal with special Scroll
  if s in ('skopt', 'skopt_bogp'):
    # Delay this importing
    from .skopt_bo.bogp import Bayesian
    return Bayesian

  # Deal with normal Scroll
  if not s in SCROLL_SHELF:
    raise KeyError("!! Unknown scroll name '{}'".format(s))
  return SCROLL_SHELF[s]


def get_argument_keys():
  # Get all sub-classes of Scroll
  classes = list(set(SCROLL_SHELF.values())) + [Scroll]
  keys = []
  for c in classes: keys.extend(
    [k for k in c.__init__.__code__.co_varnames if k not in
     ('self', 'hyper_params', 'constraints', 'observation_fetcher', 'kwargs')])
  return list(set(keys))

from .scroll_base import Scroll
from .goose import Goose
from .grid_search import GridSearch


SCROLL_SHELF = {
  'grid': GridSearch,
  'grid_search': GridSearch,
  'goose': Goose,
}


def get_scroll(s):
  if isinstance(s, Scroll): return s
  assert isinstance(s, str)
  s = s.lower().replace('-', '_')
  if not s in SCROLL_SHELF:
    raise KeyError("!! Unknown scroll name '{}'".format(s))
  return SCROLL_SHELF[s]
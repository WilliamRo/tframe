from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


WISDOMS_WITH_QUOTES = (
  ('Believe and act as if it were impossible to fail.',
   'Charles F. Kettering'),
  ('Guard well your thoughts when alone and your words when accompanied',
   'Roy T. Bennett'),
  ("Courage is't having the strength to go on. "
   "It's going on when you don't have strength.",
   'Napoleon'),
  ("A true hero isn't measured by the size of his strength, "
   "but by the strength of his heart",
   '<Hercules>'),
  ("The greater the obstacle, the more glory in overcoming it.",
   "Moliere"),
  ("Freedom lies in being bold.",
   "Robert Frost"),
  ("You've got to get up every morning with determination if you're going"
   " to go to bed with satisfaction",
   "George Horace Lorimer"),
  ("He who is not everyday conquering some fear has not learned the secret of "
   "life.",
   "Emerson"),
)


def rand_wisdom():
  wisdom, quote = WISDOMS_WITH_QUOTES[
    np.random.randint(len(WISDOMS_WITH_QUOTES))]
  return '{}    — {}'.format(wisdom, quote)



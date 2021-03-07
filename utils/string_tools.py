from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


def merger(str_list):
  assert isinstance(str_list, list) and len(str_list) > 0
  counter = 0
  current = None
  merged = []

  def archive():
    if current is None: return
    s = current
    if counter > 1: s = '({})x{}'.format(s, counter)
    merged.append(s)

  for i, string in enumerate(str_list):
    if string != current:
      # Append previous string(s) to merged list
      archive()
      # Set new string
      current = string
      counter = 1
    else: counter += 1
    if i == len(str_list) - 1: archive()

  return merged


def get_time_string():
  return time.strftime('[{}-{}-%d %H:%M:%S]'.format(
    time.strftime('%Y')[2:], time.strftime('%B')[:3]))


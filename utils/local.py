from __future__ import absolute_import

import os
import re


def check_path(*paths):
  assert len(paths) > 0
  if len(paths) == 1:
    paths = re.split(r'/|\\', paths[0])
    if paths[0] in ['.', '']:
      paths.pop(0)
    if paths[-1] == '':
      paths.pop(-1)
  path = ""
  for p in paths:
    path = os.path.join(path, p)
    if not os.path.exists(path):
      os.mkdir(path)

  return path


def clear_paths(paths):
  for path in paths:
    # Delete all files in path
    for root, dirs, files in os.walk(path, topdown=False):
      for file in files:
        os.remove(os.path.join(root, file))



from __future__ import absolute_import

import os
import re
import six

from . import console


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
  if isinstance(paths, six.string_types):
    paths = [paths]

  for path in paths:
    # Delete all files in path
    for root, dirs, files in os.walk(path, topdown=False):
      # Remove directories
      for folder in dirs:
        clear_paths(os.path.join(root, folder))
      # Delete files
      for file in files:
        os.remove(os.path.join(root, file))

    # Show status
    console.show_status('Directory "{}" has been cleared'.format(path))



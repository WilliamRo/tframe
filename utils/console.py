from __future__ import absolute_import

import pprint
import time
import os

from sys import stdout


_config = {
  'default_line_width': 80,
  'default_title': 'main',
  'prompt': '>>',
  'sub_prompt': '...',
  'tail_width': 12
}
_config['bar_width'] = _config['default_line_width'] - _config['tail_width'] - 2

_cache = {}
_pp = pprint.PrettyPrinter()


def start(title=None, width=None):
  title = title or _config['default_title']
  width = width or _config['default_line_width']
  print("-> Start of %s\n%s" % (title, '-' * width))
  _cache["title"] = title


def end(width=None):
  title = _cache.pop("title", _config['default_title'])
  width = width or _config['default_line_width']
  print("%s\n|> End of %s" % (('-' * width), title))


def section(contents):
  print("=" * _config['default_line_width'])
  print(":: %s" %  contents)
  print("=" * _config['default_line_width'])


def show_status(content):
  print("%s %s" % (_config['prompt'], content))


def supplement(content, level=1):
  print("{} {}".format(level * _config['sub_prompt'], content))


def pprint(content):
  _pp.pprint(content)


def print_progress(index=None, total=None, start_time=None, progress=None):
  """
  Print progress bar, the line which cursor is positioned will be overwritten
  
  :param index: positive scalar, indicating current work progress
  :param total: positive scalar, indicating total work 
  :param start_time: if provided, ETA will be displayed to the right of
                      the progress bar
  :param progress: ...
  """
  if progress is None:
    if index is None or total is None:
      raise ValueError('index and total must be provided')
    progress = 1.0 * index / total

  if start_time is not None:
    duration = time.time() - start_time
    eta = duration / max(progress, 1e-7) * (1 - progress)
    tail = "ETA: {:.0f}s".format(eta)
  else:
    tail = "{:.0f}%".format(100 * progress)

  left = int(progress * _config['bar_width'])
  right = _config['bar_width'] - left
  mid = '=' if progress == 1 else '>'
  stdout.write('[%s%s%s] %s' %
               ('=' * left, mid, ' ' * right, tail))
  stdout.flush()


def write_line(content):
  stdout.write("\r{}\n".format(content))
  stdout.flush()


def clear_line():
  stdout.write("\r{}\r".format(" " * (_config['bar_width'] +
                                      _config['tail_width'])))
  stdout.flush()


def set_logging_level(level):
  """
  Set tensorflow logging level 
  :param level: integer \in {0, 1, 2, 3}
  """
  assert 0 <= level <= 3
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '{}'.format(level)


def suppress_logging():
  set_logging_level(2)


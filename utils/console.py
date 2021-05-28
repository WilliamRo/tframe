from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps
import pprint as _pprint
import time
import os

from sys import stdout
from tframe import tf

from tframe.utils import misc


_config = {
  'default_line_width': 80,
  'default_title': 'main',
  'status_prompt': '>>',
  'warning': ' !',
  'error': '!!',
  'sub_prompt': '..',
  'tail_width': 15
}
_config['bar_width'] = _config['default_line_width'] - _config['tail_width'] - 2

_cache = {'last_called': None}
_pp = _pprint.PrettyPrinter()


def auto_clear(meth):
  @wraps(meth)
  def wrapper(*args, **kwargs):
    if _cache['last_called'] is print_progress:
      clear_line()
    _cache['last_called'] = meth
    return meth(*args, **kwargs)
  return wrapper


def clear_line():
  stdout.write("\r{}\r".format(" " * (_config['bar_width'] +
                                      _config['tail_width'])))
  stdout.flush()


@auto_clear
def start(title=None, width=None):
  title = title or _config['default_title']
  width = width or _config['default_line_width']
  print("-> Start of %s\n%s" % (title, '-' * width))
  _cache["title"] = title


@auto_clear
def end(width=None):
  title = _cache.pop("title", _config['default_title'])
  width = width or _config['default_line_width']
  print("%s\n|> End of %s" % (('-' * width), title))


@auto_clear
def section(contents):
  print("=" * _config['default_line_width'])
  print(":: %s" %  contents)
  print("=" * _config['default_line_width'])


@auto_clear
def show_status(content, symbol=_config['status_prompt']):
  print("%s %s" % (symbol, content))

@auto_clear
def show_info(info, symbol='::'):
  print("%s %s" % (symbol, info))


@auto_clear
def warning(content):
  print("%s %s" % (_config['warning'], content))


@auto_clear
def error(content):
  print("%s %s" % (_config['error'], content))


@auto_clear
def supplement(content, level=1):
  print("{} {}".format(level * _config['sub_prompt'], content))


@auto_clear
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
  progress = min(progress, 1.0)

  if start_time is not None:
    duration = time.time() - start_time
    eta = duration / max(progress, 1e-7) * (1 - progress)
    tail = "ETA: {:.0f}s".format(eta)
  else:
    tail = "{:.0f}%".format(100 * progress)

  left = int(progress * _config['bar_width'])
  right = _config['bar_width'] - left
  mid = '=' if progress == 1 else '>'
  clear_line()
  stdout.write('[%s%s%s] %s' %
               ('=' * left, mid, ' ' * right, tail))
  stdout.flush()
  _cache['last_called'] = print_progress


def split(splitter='-'):
  num = int(79 / len(splitter))
  print(num * splitter)


@auto_clear
def write_line(content):
  stdout.write("\r{}\n".format(content))
  stdout.flush()


def write(content):
  stdout.write(content)
  stdout.flush()


def set_logging_level(level):
  """
  Set tensorflow logging level 
  :param level: integer \in {0, 1, 2, 3}
  """
  assert 0 <= level <= 3
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '{}'.format(level)


def suppress_logging():
  import logging
  logging.getLogger('tensorflow').disabled = True
  set_logging_level(2)


def execute_py(path, **kwargs):
  os.system('python {} {}'.format(path, ' '.join(
    ['--{} {}'.format(k, kwargs[k]) for k in kwargs.keys()])))


def read(msg, with_enter=True):
  assert isinstance(msg, str)
  if with_enter:
    return input(msg)
  else:
    import msvcrt
    write(msg)
    char = str(msvcrt.getch())[2]
    print()
    return char


def eval_show(tensor, name=None, feed_dict=None):
  if name is None: name = misc.retrieve_name(tensor)
  sess = tf.get_default_session()
  val = sess.run(tensor, feed_dict=feed_dict)
  if len(val.shape) > 1:
    show_status('{} = '.format(name))
    pprint(val)
  else: show_status('{} = {}'.format(name, val))
  return val


def warning_with_pause(msg):
  print(' ! {}'.format(msg))
  print('.. Input `q` to quit, and any others to continue:')
  if input() == 'q': exit()


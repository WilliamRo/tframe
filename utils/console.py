import pprint
import time

from sys import stdout


_default_length = 80
_default_title = 'main'
_prompt = '>>'
_sub_prompt = "..."
_pp = pprint.PrettyPrinter()
_tail_length = 12
_bar_length =_default_length - _tail_length - 2

_cache = {}


def start(title=None, length=None):
  title = title or _default_title
  length = length or _default_length
  print("-> Start of %s\n%s" % (title, '-' * length))
  _cache["title"] = title


def end(length=None):
  title = _cache.pop("title", _default_title)
  length = length or _default_length
  print("%s\n|> End of %s" % (('-' * length), title))


def section(contents):
  print("=" * _default_length)
  print(":: %s" %  contents)
  print("=" * _default_length)


def state(content):
  print("%s %s" % (_prompt, content))


def supplement(content):
  print("{} {}".format(_sub_prompt, content))


def pprint(content):
  _pp.pprint(content)


def print_progress(index, total, start_time=None):
  if start_time is not None:
    duration = time.time() - start_time
    eta = duration / index * (total - index)
    tail = "ETA: {:.0f}s".format(eta)
  else:
    tail = "{:.0f}%".format(100 * index / total)

  left = int(index * _bar_length / total)
  right = _bar_length - left
  mid = '=' if index == total else '>'
  stdout.write('[%s%s%s] %s' %
               ('=' * left, mid, ' ' * right, tail))
  stdout.flush()


def write_line(content):
  stdout.write("\r{}\n".format(content))
  stdout.flush()


def clear_line():
  stdout.write("\r{}\r".format(" " * (_bar_length + _tail_length)))
  stdout.flush()


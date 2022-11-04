from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import six

from tframe import tf

import tframe as tfr

from . import console
from fnmatch import fnmatch


def check_path(*paths, create_path=True, is_file_path=False):
  assert len(paths) > 0
  if len(paths) == 1:
    paths = re.split(r'/|\\', paths[0])
    if paths[0] in ['.', '']:
      paths.pop(0)
    if len(paths) > 0 and paths[-1] == '':
      paths.pop(-1)
  path = ""
  for i, p in enumerate(paths):
    # The first p should be treated differently
    if i == 0:
      assert path == "" and p != ""
      # if p[-1] != ':':
      if ':' not in p:
        # Put `/` back to front for Unix-like systems
        path = '/' + p
      else:
        # This will only happen in Windows system family
        path = p + '\\'
        # continue
    else:
      path = os.path.join(path, p)

    # Make directory if necessary
    if not (is_file_path and i == len(paths) - 1):
      if not os.path.exists(path):
        # TODO: flag in context.hub should not be used here
        if tfr.context.hub.should_create_path and create_path:
          os.mkdir(path)
        else:
          raise AssertionError('!! Directory {} does not exist'.format(path))
  return path


def clear_paths(paths):
  if len(paths) == 0: return
  if isinstance(paths, six.string_types):
    paths = [paths]

  console.show_status('Cleaning path ...')
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
    console.supplement('Directory "{}" has been cleared'.format(path))


def load_checkpoint(path, session, saver):
  console.show_status("Access to directory '{}' ...".format(path))
  ckpt_state = tf.train.get_checkpoint_state(path)

  if ckpt_state and ckpt_state.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt_state.model_checkpoint_path)
    saver.restore(session, os.path.join(path, ckpt_name))
    # Find counter
    step_list = re.findall(r'-(\d+)$', ckpt_name)
    assert len(step_list) == 1
    counter = int(step_list[0])
    # Try to find rounds
    rnd_list = re.findall(r'\((\d+.\d+)_rounds\)', ckpt_name)
    if len(rnd_list) > 0:
      assert len(rnd_list) == 1
      rnd = float(rnd_list[0])
    else: rnd = None
    # Show status
    console.show_status("Loaded {}".format(ckpt_name))
    return True, counter, rnd
  else:
    if tfr.context.hub.train and tfr.context.hub.save_model:
      console.show_status('New checkpoints will be created ...')
    else:
      console.warning('Can not found model checkpoint')
    return False, 0, None


def save_checkpoint(path, session, saver, step):
  assert isinstance(saver, tf.train.Saver)
  saver.save(session, path, step)


def write_file(path, content, append=False):
  mode = 'a' if append else 'w'
  f = open(path, mode)
  f.write(content)
  f.close()


def wizard(pattern=None, extension=None, current_dir=None, max_depth=1,
           input_with_enter=True):
  assert isinstance(max_depth, int) and max_depth >= 0
  assert (isinstance(pattern, str) and len(pattern) > 0 or
          isinstance(extension, str) and len(extension) > 0)
  if extension is not None: pattern = r'[\w \(\),-]+\.{}'.format(extension)
  # if extension is not None: pattern = r'.+\.{}'.format(extension)

  input = lambda msg: console.read(msg, input_with_enter)

  is_file = lambda name: re.fullmatch(r'[\w \(\),-]+\.[\w]+', name) is not None
  may_be_dir = lambda name: re.fullmatch(r'[\w \(\),-]+', name) is not None
  is_target = lambda name: re.fullmatch(pattern, name) is not None
  def contain_target(dir, max_depth):
    full_path = lambda f: os.path.join(dir, f)
    for file in os.listdir(dir):
      if is_file(file):
        if is_target(file): return True
      elif (may_be_dir(file) and os.path.isdir(full_path(file)) and
            max_depth > 0 and contain_target(full_path(file), max_depth - 1)):
        return True
    return False
  def search(dir, max_depth):
    targets = []
    full_path = lambda f: os.path.join(dir, f)
    for file in os.listdir(dir):
      # if is_dir(file):
      if may_be_dir(file) and os.path.isdir(full_path(file)):
        if max_depth > 0 and contain_target(full_path(file), max_depth - 1):
          targets.append(file)
      elif is_target(file): targets.append(file)
    return targets

  if current_dir is None: current_dir = os.getcwd()
  dir_stack = []
  selected_file = None
  while selected_file is None:
    targets = search(current_dir, max_depth - len(dir_stack))
    if len(dir_stack) > 1: targets = list(reversed(targets))
    if len(targets) == 0:
      console.show_status('Can not find targets in `{}`'.format(current_dir))
      return None
    # Print targets
    console.show_status('Current directory is `{}`'.format(current_dir))
    for i, t in enumerate(targets):
      console.supplement('[{}] {}'.format(i + 1, t))
    selection = input('=> Please input: ')

    int_list = list(range(1, 26))
    str_list = [str(i) for i in range(1, 10)] + list('abcdefghijklmnop')
    def get_str(i): return {i: s for i, s in zip(int_list, str_list)}[i]
    def get_int(s): return {s: i for i, s in zip(int_list, str_list)}[s]
    while True:
      if selection in ('..', '0') and len(dir_stack) > 0:
        current_dir = dir_stack.pop(-1)
        break
      elif selection == 'q': quit()
      elif selection in [get_str(i + 1) for i in range(len(targets))]:
        file = targets[get_int(selection) - 1]
        if is_target(file):
          selected_file = os.path.join(current_dir, file)
        else:
          dir_stack.append(current_dir)
          path = os.path.join(current_dir, file)
          current_dir = path
        break
      else:
        selection = input(
        '=> Invalid input `{}`, please input again: '.format(selection))

  return selected_file


def load_wav_file(file_name, use_librosa=False, sr=None):
  if use_librosa:
    import librosa
    if sr is None: sr = 22050
    signal_, fs = librosa.core.load(file_name, sr)
  else:
    import scipy.io.wavfile as wavfile
    fs, signal_ = wavfile.read(file_name)
  return signal_, fs


def re_find_single(pattern, file_name=None):
  assert isinstance(pattern, str) and len(pattern) > 0
  if file_name is None:
    import __main__
    if not hasattr(__main__, '__file__'):
      raise AttributeError('!! __main__ does not have attribute `__file__`')
    file_name = __main__.__file__
  matched = re.findall(pattern, file_name)
  if len(matched) == 0: raise ValueError(
    '!! no substring matched pattern `{}`'.format(pattern))
  if len(matched) > 1: raise ValueError(
    '!! found more than 1 match of pattern `{}`'.format(pattern))
  return matched[0]


def walk(root_path, type_filter=None, pattern=None, return_basename=False):
  """Find all required contents under the given path"""
  # Sanity check
  if not os.path.exists(root_path):
    raise FileNotFoundError('!! `{}` not exist'.format(root_path))
  paths = [os.path.join(root_path, p) for p in os.listdir(root_path)]
  # Filter path
  if type_filter in ('file',): type_filter = os.path.isfile
  elif type_filter in ('folder', 'dir'): type_filter = os.path.isdir
  else: assert type_filter is None
  if callable(type_filter): paths = list(filter(type_filter, paths))
  # Filter pattern
  if pattern is not None:
    paths = list(filter(lambda p: fnmatch(p, pattern), paths))
  if return_basename: return [os.path.basename(p) for p in paths]
  return paths



from __future__ import absolute_import

import os
import re
import six

import tensorflow as tf

import tframe as tfr
from . import console


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
      if p[-1] != ':':
        # Put `/` back to front for Unix-like systems
        path = '/' + p
      else:
        # This will only happen in Windows system family
        path = p
        continue
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
    counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
    console.show_status("Loaded {}".format(ckpt_name))
    return True, counter
  else:
    if tfr.context.hub.train and tfr.context.hub.save_model:
      console.show_status('New checkpoints will be created ...')
    else:
      console.warning('Can not found model checkpoint')
    return False, 0


def save_checkpoint(path, session, saver, step):
  assert isinstance(saver, tf.train.Saver)
  saver.save(session, path, step)


def write_file(path, content, append=False):
  mode = 'a' if append else 'w'
  f = open(path, mode)
  f.write(content)
  f.close()


def wizard(extension, current_dir=None, max_depth=1, input_with_enter=True):
  assert isinstance(max_depth, int) and max_depth >= 0
  assert isinstance(extension, str) and len(extension) > 0

  input = lambda msg: console.read(msg, input_with_enter)

  # targets = []
  is_file = lambda name: '.' in name
  is_target = lambda name: is_file(name) and name.split('.')[-1] == extension
  def contain_target(dir, max_depth):
    full_path = lambda f: os.path.join(dir, f)
    for file in os.listdir(dir):
      if is_file(file):
        if is_target(file): return True
      elif max_depth > 0 and contain_target(full_path(file), max_depth - 1):
        return True
    return False
  def search(dir, max_depth):
    targets = []
    full_path = lambda f: os.path.join(dir, f)
    for file in os.listdir(dir):
      if not is_file(file):
        if max_depth > 0 and contain_target(full_path(file), max_depth - 1):
          targets.append(file)
      elif is_target(file): targets.append(file)
    return targets

  if current_dir is None: current_dir = os.getcwd()
  dir_stack = []
  selected_file = None
  while selected_file is None:
    targets = search(current_dir, max_depth - len(dir_stack))
    if len(targets) == 0:
      console.show_status('Can not find targets in `{}`'.format(current_dir))
      return None
    # Print targets
    console.show_status('Current directory is `{}`'.format(current_dir))
    for i, t in enumerate(targets):
      console.supplement('[{}] {}'.format(i + 1, t))
    selection = input('=> Please input: ')
    while True:
      if selection in ('..', '0') and len(dir_stack) > 0:
        current_dir = dir_stack.pop(-1)
        break
      elif selection == 'q': quit()
      elif selection in [str(i + 1) for i in range(len(targets))]:
        file = targets[int(selection) - 1]
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


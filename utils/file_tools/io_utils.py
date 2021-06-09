# ROMA

import os
import pickle
import time


def check_path(*path, target_is_file=False, create_if_not_exist=True):
  """Make sure that the corresponding file/dir exists.
  If target is a file, new directory will be created if 'create_if_not_exist'
  flag is True. Otherwise an error will be raised.

  :param path: paths to be joined as a target path
  :param target_is_file: whether the target is a file
  :param create_if_not_exist: option to create directory if input directory does
                              not exist
  :return: the input path
  """
  if target_is_file: create_if_not_exist = False

  p = os.path.join(*path)
  if not os.path.exists(p):
    if create_if_not_exist: os.makedirs(p)
    else: raise FileExistsError('!! {} {} does not exist.'.format(
      'File' if target_is_file else 'Directory', p))
  return p


def safe_open(p, mode, wait_time=0.1, time_out=100):
  """Safely open a file for reading or writing.

  :param p: first argument for open()
  :param mode: second argument for open()
  :param wait_time: waiting time before retry
  :param time_out: maximum waiting time
  :return: file handler returned by open
  """
  if 'r' in mode and not os.path.exists(p):
    raise FileExistsError('!! File `{}` not exist'.format(p))
  tic = time.time()
  while True:
    try:
      return open(p, mode)
    except:
      time.sleep(wait_time)
    # Check if time is out
    if time.time() - tic > time_out:
      raise TimeoutError('!! Failed to open `{}`, time out.'.format(p))


def load(*path):
  """Load file using pickle"""
  p = check_path(*path, target_is_file=True)
  with safe_open(p, 'rb') as f:
    return pickle.load(f)


def save(obj, *path, protocol=pickle.HIGHEST_PROTOCOL):
  """Safely write file to file system. If file is occupied, this function will
     wait a bit and retry.

  :param obj: Python object to be dumped
  :param path: path to save obj, will be joined to form a path
  :param protocol: third argument for pickle.dump
  """
  p = os.path.join(*path)
  with safe_open(p, 'wb') as f:
    pickle.dump(obj, f, protocol)


if __name__ == '__main__':
  from os.path import expanduser
  import numpy as np
  from roma import console

  home = expanduser('~')

  # region: safe_dump test

  dir_path = check_path(home, 'roma-tmp', 'safe_dump_test')
  file_name = 'test.sum'
  p = os.path.join(dir_path, file_name)

  # endregion: safe_dump test

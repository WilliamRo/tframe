from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import traceback



def update_job_dir(id, model_name, fs_index=-2):
  """In talos (tframe) convention, say we have a default project structure:
     -----------------------------------------
     DEPTH  0          1         2 (*)
            this_proj
                    |- 01-MNIST
                              |- mn_core.py
                              |- mn_du.py
                              |- mn_mu.py
                              |- t1_lenet.py
                    |- 02-CIFAR10
                    |- ...
                    |- tframe
     -----------------------------------------
     Then for t1_lenet.py, th.job_dir should be 'this_proj/01-MNIST/01_lenet',
     here
      - task_dir = 'this_proj/01-MNIST' (dir_name of abs_path(t1_lenet.py))
      - job_fn = '01_lenet',
  """
  from tframe import hub as th
  from tframe.utils.local import check_path

  dirname = os.path.dirname
  # Job folder name
  job_fn = '{:02d}_{}'.format(id, model_name)
  file_stack = [check_path(os.path.abspath(pkg[0]))
                for pkg in list(traceback.extract_stack())]
  task_dir = dirname(file_stack[fs_index])

  # Scaffold
  # for i, s in enumerate(file_stack):
  #   print(i, s)
  # assert False

  # Case (1): running t-file located in checkpoints model
  if os.path.join(job_fn, 'checkpoints') in task_dir:
    th.job_dir = dirname(dirname(task_dir))
    return

  # Case (2): running s-file
  if th.job_dir not in task_dir:
    # Find correct job_dir for algorithms such as skopt
    for fn in file_stack[::-1]:
      if th.job_dir in fn:
        th.job_dir = dirname(fn)
        return
    raise AssertionError('!! failed to extract job_dir while running s-file')

  # Case (3): running t-file from common location, `task_dir` should
  # be the folder containing this module file
  th.job_dir = os.path.join(task_dir, job_fn)

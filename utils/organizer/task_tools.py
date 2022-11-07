from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import traceback



def update_job_dir(id, model_name):
  from tframe import hub as th
  from tframe.utils.local import check_path

  dirname = os.path.dirname
  # Job folder name
  job_fn = '{:02d}_{}'.format(id, model_name)
  file_stack = [check_path(pkg[0]) for pkg in list(traceback.extract_stack())]
  task_dir = dirname(file_stack[-2])

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

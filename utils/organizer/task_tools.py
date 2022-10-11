from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import traceback



def update_job_dir(id, model_name):
  from tframe import hub as th

  # th.job_dir += '/{:02d}_{}'.format(id, model_name)

  file, _, _, _ = list(traceback.extract_stack())[-2]
  task_dir = os.path.dirname(file)

  # TODO: the patch below is used currently
  job_fn = '{:02d}_{}'.format(id, model_name)
  if any([f'{job_fn}{c}checkpoints' in task_dir for c in ('/', '\\')]):
    # For trained model, the task module is located inside the
    # corresponding checkpoints folder,
    th.job_dir = os.path.dirname(os.path.dirname(task_dir))
  else:
    # While running normal task module, `task_dir` should be the folder
    # containing this module file
    th.job_dir = os.path.join(task_dir, job_fn)


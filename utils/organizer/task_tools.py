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
  th.job_dir = os.path.join(task_dir, '{:02d}_{}'.format(id, model_name))


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def update_job_dir(id, model_name):
  from tframe import hub as th
  th.job_dir += '/{:02d}_{}'.format(id, model_name)

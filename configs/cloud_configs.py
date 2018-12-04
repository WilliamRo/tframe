from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class CloudConfigs(object):
  """Configurations typically for deploying on GCP"""

  on_cloud = Flag.boolean(
    False, 'Whether this task is running on the cloud',
    name='cloud')
  hp_tuning = Flag.boolean(
    False, 'Whether this is a hyper-parameter tuning task',
    name='hpt')

  def smooth_out_cloud_configs(self):
    assert hasattr(self, 'job_dir') and hasattr(self, 'train')

    if '://' in self.job_dir: self.on_cloud = True
    if self.on_cloud or self.hp_tuning:
      self.export_note = False
      self.progress_bar = False
    if self.on_cloud:
      self.snapshot = False
      self.monitor = False
    if self.hp_tuning:
      self.summary = False
      self.save_model = False
    if not self.train and self.on_cloud: self.overwrite = False

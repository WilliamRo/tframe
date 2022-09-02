from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .flag import Flag


class MonitorConfigs(object):

  # monitor = Flag.boolean(None, 'Whether to monitor or not (of highest '
  #                              'priority)')
  # monitor_grad = Flag.boolean(False, 'Whether to monitor gradients or not')
  # monitor_weight = Flag.boolean(False, 'Whether to monitor weights or not')
  # monitor_preact = Flag.boolean(False, 'Whether to enable pre-act summary')
  # monitor_postact = Flag.boolean(False, 'Whether to enable post-act summary')
  monitor_weight_grads = Flag.boolean(False, 'Whether to monitor weights grad')
  monitor_weight_flips = Flag.boolean(False, 'Whether to monitor weights flips')
  monitor_weight_history = Flag.boolean(False, 'Whether to monitor weights '
                                               'history')

  def smooth_out_monitor_configs(self):
    if self.monitor_weight_flips:
      self.monitor_weight_history = True

    # if self.monitor in (True, False):
    #   self.monitor_grad = self.monitor
    #   self.monitor_weight = self.monitor
      # self.monitor_preact = self.monitor
      # self.monitor_postact = self.monitor


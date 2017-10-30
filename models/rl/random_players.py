from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe.models.rl.interfaces import Player
from tframe.models.rl.interfaces import FMDPAgent


class FMDRandomPlayer(Player):

  def next_step(self, agent):
    assert isinstance(agent, FMDPAgent)

    candidates = agent.candidate_states
    action_index = np.random.randint(0, len(candidates))
    reward = agent.act(action_index)

    return reward
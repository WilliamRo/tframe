from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Agent(object):

  @property
  def state(self):
    raise NotImplementedError('Property state not implemented')

  @property
  def candidate_states(self):
    raise NotImplementedError('Property candidate_states not implemented')

  def act(self, action):
    raise NotImplementedError('Method act not implemented')

  def action_index(self, values):
    raise NotImplementedError('Method action_index not implemented')


class FiniteMarkovDecisionProcessAgent(Agent):

  def restart(self):
    raise NotImplementedError('Method restart not implemented')

  def compete(self, players, rounds, **kwargs):
    raise NotImplementedError('Method compete not implemented')

  @property
  def terminated(self):
    raise NotImplementedError('Property terminated not implemented')


class Player(object):

  def next_step(self, agent):
    raise NotImplementedError('Method next_step not implemented')


# Alias
FMDPAgent = FiniteMarkovDecisionProcessAgent



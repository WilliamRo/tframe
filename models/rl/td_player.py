from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import tensorflow as tf

from tframe.models.feedforward import Feedforward

from tframe import console
from tframe import pedia
from tframe import FLAGS
from tframe import with_graph

from tframe.models.rl.interfaces import FMDPAgent
from tframe.models.rl.interfaces import Player
from tframe.models.rl.random_players import FMDRandomPlayer


class TDPlayer(Feedforward, Player):
  """Estimator for Markov Decision Process (MDP) with finite states and 
     actions."""
  def __init__(self, mark=None):
    Feedforward.__init__(self, mark)
    self._next_value = None
    self._update_op = None

    self._opponent = None

  # region : Properties

  @property
  def description(self):
    return self.structure_string()

  # endregion : Properties

  # region : Build

  @with_graph
  def _build(self, lamda=0.5, learning_rate=0.01):
    Feedforward._build(self)
    # Initialize target placeholder
    self._next_value = tf.placeholder(
      self._outputs.dtype, self._outputs.get_shape(), name='next_value')

    # Define loss
    with tf.name_scope('Loss'):
      delta = tf.reduce_sum(self._next_value - self._outputs, name='delta')
      self._loss = 0.5 * tf.square(delta)
      tf.summary.scalar('loss_sum', self._loss)

    # Define update op
    update_op = []
    with tf.variable_scope('Update_Ops'):
      # Define gradient
      with tf.name_scope('Gradients'):
        vars = tf.trainable_variables()
        grads = tf.gradients(self._outputs, vars)
      # Update model with eligibility traces
      for var, grad in zip(vars, grads):
        with tf.variable_scope('trace'):
          trace = tf.Variable(
            tf.zeros(grad.get_shape()), trainable=False, name='trace')
          trace_op = trace.assign((lamda * trace) + grad)
          # TODO: add histogram here

          grad_trace = learning_rate * delta * trace_op
          grad_apply = var.assign_add(grad_trace)
          update_op.append(grad_apply)
      # Group ops into a single op
      self._update_op = tf.group(*update_op, name='train')

    # Print status and model structure
    self._show_building_info(FeedforwardNet=self)

    # Launch session
    self.launch_model(FLAGS.overwrite and FLAGS.train)

  # endregion : Build

  # region : Train

  def train(self, agent, episodes=500, print_cycle=0, snapshot_cycle=0,
             match_cycle=0, rounds=100, rate_thresh=1.0, shadow=None,
             save_cycle=100, snapshot_function=None):
    # Validate agent
    if not isinstance(agent, FMDPAgent):
      raise TypeError('Agent should be a FMDP-agent')

    # Check settings TODO: codes should be reused
    if snapshot_function is not None:
      if not callable(snapshot_function):
        raise ValueError('snapshot_function must be callable')
      self._snapshot_function = snapshot_function

    print_cycle = FLAGS.print_cycle if FLAGS.print_cycle >= 0 else print_cycle
    snapshot_cycle = (FLAGS.snapshot_cycle if FLAGS.snapshot_cycle >= 0
                      else snapshot_cycle)
    match_cycle = FLAGS.match_cycle if FLAGS.match_cycle >= 0 else match_cycle

    # Show configurations
    console.show_status('Configurations:')
    console.supplement('episodes: {}'.format(episodes))

    # Do some preparation
    if self._session is None:
      self.launch_model()

    assert isinstance(self._graph, tf.Graph)
    with self._graph.as_default():
      if self._merged_summary is None:
        self._merged_summary = tf.summary.merge_all()

    # Set opponent
    if match_cycle > 0:
      if shadow is None:
        self._opponent = FMDRandomPlayer()
        self._opponent.player_name = 'Random Player'
      elif isinstance(shadow, TDPlayer):
        self._opponent = shadow
        self._opponent.player_name = 'Shadow_{}'.format(self._opponent.counter)
      else:
        raise TypeError('Opponent should be an instance of TDPlayer')

    # Begin training iteration
    assert isinstance(agent, FMDPAgent)
    console.section('Begin episodes')
    for epi in range(1, episodes + 1):
      # Initialize variable
      agent.restart()
      if hasattr(agent, 'default_first_move'):
        agent.default_first_move()
      # Record episode start time
      start_time = time.time()
      steps = 0

      state = agent.state
      summary = None
      # Begin current episode
      while not agent.terminated:
        # Make a move
        next_value = self.next_step(agent)
        steps += 1
        # Update model
        state = np.reshape(state, (1,) + state.shape)
        next_value = np.reshape(np.array(next_value), (1, 1))
        feed_dict = {self.input_[0]: state, self._next_value: next_value}
        feed_dict.update(self._get_status_feed_dict(is_training=True))
        assert isinstance(self._session, tf.Session)
        summary, _ = self._session.run(
          [self._merged_summary, self._update_op], feed_dict)

        state = agent.state

      # End of current episode
      self.counter += 1

      assert isinstance(self._summary_writer, tf.summary.FileWriter)
      self._summary_writer.add_summary(summary, self.counter)

      if print_cycle > 0 and np.mod(self.counter, print_cycle) == 0:
        self._print_progress(epi, start_time, steps, total=episodes)
      if snapshot_cycle > 0 and np.mod(self.counter, snapshot_cycle) == 0:
        self._snapshot(epi / episodes)
      if match_cycle > 0 and np.mod(self.counter, match_cycle) == 0:
        self._training_match(agent, rounds, epi / episodes, rate_thresh)
      if np.mod(self.counter, save_cycle) == 0:
        self._save(self.counter)

    # End training
    console.clear_line()
    self._summary_writer.flush()
    self.shutdown()

  def _print_progress(self, epi, start_time, steps, **kwargs):
    """Use a awkward way to avoid IDE warning :("""
    console.clear_line()
    console.show_status(
      'Episode {} [{} total] {} steps, Time elapsed = {:.2f} sec'.format(
        epi, self.counter, steps, time.time() - start_time))
    console.print_progress(epi, kwargs.get('total'))

  def _snapshot(self, progress):
    if self._snapshot_function is None:
      return

    filename = 'train_{}_episode'.format(self.counter)
    fullname = "{}/{}".format(self.snapshot_dir, filename)
    self._snapshot_function(fullname)

    console.clear_line()
    console.write_line("[Snapshot] snapshot saved to {}".format(filename))
    console.print_progress(progress=progress)

  def _training_match(self, agent, rounds, progress, rate_thresh):
    # TODO: inference graph is hard to build under this frame => compromise
    if self._opponent is None:
      return
    assert isinstance(agent, FMDPAgent)

    console.clear_line()
    title = 'Training match with {}'.format(self._opponent.player_name)
    rate = self.compete(agent, rounds, self._opponent, title=title)
    if rate >= rate_thresh and isinstance(self._opponent, TDPlayer):
      # Find an stronger opponent
      self._opponent._load()
      self._opponent.player_name = 'Shadow_{}'.format(self._opponent.counter)
      console.show_status('Opponent updated')

    console.print_progress(progress=progress)

  # endregion : Train

  # region : Public Methods

  def compete(self, agent, rounds, opponent, title='Competition'):
    console.show_status('[{}]'.format(title))
    assert isinstance(agent, FMDPAgent)
    rate, reports = agent.compete([self, opponent], rounds)
    for report in reports:
      console.supplement(report)
    return rate

  def next_step(self, agent):
    assert agent is not None and isinstance(agent, FMDPAgent)

    candidates = agent.candidate_states
    values = self.estimate(candidates)
    action_index = agent.action_index(values)
    reward = agent.act(action_index)
    return reward if agent.terminated else values[action_index]

  def estimate(self, states):
    if self._outputs is None:
      raise ValueError('Model not built yet')
    if self._session is None:
      self.launch_model(overwrite=False)

    feed_dict = {self.input_[0]: states}
    feed_dict.update(self._get_status_feed_dict(False))

    outputs = self._session.run(self._outputs, feed_dict)

    return outputs

  # endregion : Public Methods

  # region : Private Methods

  # endregion : Private Methods

  '''For some reason, do not remove this line'''

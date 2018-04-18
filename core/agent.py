from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tframe as tfr

from tframe import config
from tframe import console
from tframe import pedia

from tframe.utils.local import check_path, clear_paths, write_file
from tframe.utils.local import save_checkpoint, load_checkpoint


class Agent(object):
  """An Agent works for TFrame Model, handling tensorflow stuffs"""
  def __init__(self, model, graph=None):
    # Each agent works on one tensorflow graph with a tensorflow session
    # .. set to it
    assert isinstance(model, tfr.models.Model)
    self._model = model
    self._graph = None
    self._session = None
    self._init_session_and_graph(graph)
    # Graph variables
    self._is_training = None
    self._init_graph_variable()
    # An agent saves model and writes summary
    self._saver = tf.train.Saver()
    self._summary_writer = tf.summary.FileWriter(self.log_dir)

  # region : Properties

  # region : Accessors

  @property
  def graph(self):
    assert isinstance(self._graph, tf.Graph)
    return self._graph

  @property
  def session(self):
    assert isinstance(self._session, tf.Session)
    return self._session

  @property
  def summary_writer(self):
    return self._summary_writer

  # endregion : Accessors

  # region : Paths

  @property
  def log_dir(self):
    return check_path(config.job_dir, config.record_dir,
                      config.log_folder_name, self._model.mark)
  @property
  def ckpt_dir(self):
    return check_path(config.job_dir, config.record_dir,
                      config.ckpt_folder_name, self._model.mark)
  @property
  def snapshot_dir(self):
    return check_path(config.job_dir, config.record_dir,
                      config.snapshot_folder_name, self._model.mark)
  @property
  def model_path(self):
    return os.path.join(
      self.ckpt_dir, '{}.model'.format(self._model.model_name))

  # endregion : Paths

  # endregion : Properties

  # region : Public Methods

  def get_status_feed_dict(self, is_training):
    assert isinstance(is_training, bool)
    feed_dict = {self._is_training: is_training}
    return feed_dict

  def load(self):
    return load_checkpoint(self.ckpt_dir, self.session, self._saver)

  def save(self, step):
    save_checkpoint(self.model_path, self.session, self._saver, step)

  def launch_model(self, overwrite=False):
    config.smooth_out_conflicts()
    # Before launch session, do some cleaning work
    if overwrite and config.overwrite:
      paths = []
      if config.summary: paths.append(self.log_dir)
      if config.save_model: paths.append(self.ckpt_dir)
      if config.snapshot: paths.append(self.snapshot_dir)
      clear_paths(paths)

    # Launch session on self.graph
    console.show_status('Launching session ...')
    self._session = tf.Session(graph=self._graph)
    console.show_status('Session launched')
    # Prepare some tools
    if config.save_model: self._saver = tf.train.Saver()
    if config.summary or config.hp_tuning:
      self._summary_writer = tf.summary.FileWriter(self.log_dir)

    # Try to load exist model
    load_flag, self._model.counter = self.load()
    if not load_flag:
      assert self._model.counter == 0
      # If checkpoint does not exist, initialize all variables
      self._session.run(tf.global_variables_initializer())
      # Add graph
      if config.summary: self._summary_writer.add_graph(self._session.graph)
      # Write model description to file
      if config.snapshot:
        description_path = os.path.join(self.snapshot_dir, 'description.txt')
        write_file(description_path, self._model.description)

    return load_flag

  def shutdown(self):
    if config.summary or config.hp_tuning:
      self._summary_writer.close()
    self.session.close()

  # endregion : Public Methods

  # region : Private Methods

  def _init_session_and_graph(self, graph):
    if graph is not None:
      assert isinstance(graph, tf.Graph)
      self._graph = graph
    else: self._graph = tf.Graph()
    self._session = tf.Session(self._graph)
    # TODO
    # When linking batch-norm layer (and dropout layer),
    #   this placeholder will be got from default graph
    self._graph.is_training = self._is_training
    tfr.current_graph = self._graph

  def _init_graph_variable(self):
    with self.graph.as_default():
      self._is_training = tf.placeholder(dtype=tf.bool, name=pedia.is_training)

  # endregion : Private Methods

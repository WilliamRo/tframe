from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import tframe as tfr

from tframe import hub
from tframe import console
from tframe import pedia

from tframe.utils import imtool
from tframe.utils import Note
from tframe.utils.local import check_path, clear_paths, write_file
from tframe.utils.local import save_checkpoint, load_checkpoint

from tframe.core.decorators import with_graph


class Agent(object):
  """An Agent works for TFrame Model, handling tensorflow stuffs"""
  def __init__(self, model, graph=None):
    # Each agent works on one tensorflow graph with a tensorflow session
    # .. set to it
    assert isinstance(model, tfr.models.Model)
    self._model = model
    self._session = None
    self._graph = None
    # Graph variables
    self._is_training = None
    self._init_graph(graph)
    # An agent saves model and writes summary
    self._saver = None
    self._summary_writer = None
    # An agent holds a default note
    self._note = Note()

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
  def saver(self):
    assert isinstance(self._saver, tf.train.Saver)
    return self._saver

  @property
  def summary_writer(self):
    assert isinstance(self._summary_writer, tf.summary.FileWriter)
    return self._summary_writer

  # endregion : Accessors

  # region : Paths

  @property
  def note_dir(self):
    return check_path(hub.job_dir, hub.record_dir, hub.note_folder_name,
                      self._model.mark, create_path=hub.export_note)
  @property
  def log_dir(self):
    return check_path(hub.job_dir, hub.record_dir, hub.log_folder_name,
                      self._model.mark, create_path=hub.summary)
  @property
  def ckpt_dir(self):
    return check_path(hub.job_dir, hub.record_dir, hub.ckpt_folder_name,
                      self._model.mark, create_path=hub.save_model)
  @property
  def snapshot_dir(self):
    return check_path(hub.job_dir, hub.record_dir, hub.snapshot_folder_name,
                      self._model.mark, create_path=hub.snapshot)
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
    # TODO: when save_model option is turned off and the user want to
    #   try loading the exist model, set overwrite to False
    if not hub.save_model and hub.overwrite: return False, 0
    return load_checkpoint(self.ckpt_dir, self.session, self._saver)

  def save_model(self):
    save_checkpoint(self.model_path, self.session, self._saver,
                    self._model.counter)

  @with_graph
  def launch_model(self, overwrite=False):
    if hub.suppress_logging: console.suppress_logging()
    # Before launch session, do some cleaning work
    if overwrite and hub.overwrite:
      paths = []
      if hub.summary: paths.append(self.log_dir)
      if hub.save_model: paths.append(self.ckpt_dir)
      if hub.snapshot: paths.append(self.snapshot_dir)
      if hub.export_note: paths.append(self.note_dir)
      clear_paths(paths)
    if hub.summary: self._check_bash()

    # Launch session on self.graph
    console.show_status('Launching session ...')
    config = tf.ConfigProto()
    if not hub.allow_growth:
      value = hub.gpu_memory_fraction
      config.gpu_options.per_process_gpu_memory_fraction = value
    self._session = tf.Session(graph=self._graph, config=config)
    console.show_status('Session launched')
    # Prepare some tools
    self._saver = tf.train.Saver(var_list=self._model.variable_to_save)
    if hub.summary or hub.hp_tuning:
      self._summary_writer = tf.summary.FileWriter(self.log_dir)

    # Initialize all variables
    self._session.run(tf.global_variables_initializer())
    # Try to load exist model
    load_flag, self._model.counter = self.load()
    if not load_flag:
      assert self._model.counter == 0
      # Add graph
      if hub.summary: self._summary_writer.add_graph(self._session.graph)
      # Write model description to file
      if hub.snapshot:
        description_path = os.path.join(self.snapshot_dir, 'description.txt')
        write_file(description_path, self._model.description)
      # Show status
      console.show_status('New model initiated')

    self._model.launched = True
    self.take_notes('Model launched')
    return load_flag

  def shutdown(self):
    if hub.summary or hub.hp_tuning:
      self._summary_writer.close()
    self.session.close()

  def write_summary(self, summary, step=None):
    if not hub.summary: return
    if step is None:
      if hub.epoch_as_step and tfr.trainer.total_rounds is not None:
        step = int(tfr.trainer.total_rounds * 1000)
      else: step = self._model.counter
    assert isinstance(self._summary_writer, tf.summary.FileWriter)
    self._summary_writer.add_summary(summary, step)

  def take_notes(self, content, date_time=True, prompt=None):
    if not isinstance(content, str):
      raise TypeError('!! content must be a string')
    if isinstance(prompt, str):
      date_time = False
      content = '{} {}'.format(prompt, content)
    if date_time:
      time_str = time.strftime('[{}-{}-%d %H:%M:%S]'.format(
        time.strftime('%Y')[2:], time.strftime('%B')[:3]))
      content = '{} {}'.format(time_str, content)

    self._note.write_line(content)

  def export_notes(self, filename='notes'):
    assert hub.export_note
    file_path = '{}/{}.txt'.format(self.note_dir, filename)
    writer = open(file_path, 'a')
    writer.write('=' * 79 + '\n')
    writer.write(self._note.content + '\n')
    writer.close()
    console.show_status('Notes exported to {}'.format(file_path))

  def show_notes(self):
    console.section('Notes')
    console.write_line(self._note.content)

  def save_plot(self, fig, filename):
    imtool.save_plt(fig, '{}/{}'.format(self.snapshot_dir, filename))

  # endregion : Public Methods

  # region : Private Methods

  def _init_graph(self, graph):
    if graph is not None:
      assert isinstance(graph, tf.Graph)
      self._graph = graph
    else: self._graph = tf.Graph()
    # Initialize graph variables
    with self.graph.as_default():
      self._is_training = tf.placeholder(
        dtype=tf.bool, name=pedia.is_training)
      tf.add_to_collection(pedia.is_training, self._is_training)

    # TODO
    # When linking batch-norm layer (and dropout layer),
    #   this placeholder will be got from default graph
    # self._graph.is_training = self._is_training
    tfr.current_graph = self._graph

  def _check_bash(self):
    file_path = check_path(hub.job_dir, hub.record_dir, create_path=True)
    file_path = os.path.join(file_path, 'win_launch_tensorboard.bat')
    if not os.path.exists(file_path):
      f = open(file_path, 'w')
      f.write('tensorboard --logdir=./logs/')
      f.close()

  # endregion : Private Methods

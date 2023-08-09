from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import time

import tframe as tfr
from tframe import tf

from tframe import hub
from tframe import console
from tframe import context
from tframe import pedia
from tframe.core.nomear import Nomear
from tframe.configs.config_base import Config

from tframe.utils import imtool
from tframe.utils import Note
from tframe.utils.local import check_path, clear_paths, write_file
from tframe.utils.local import save_checkpoint, load_checkpoint
from tframe.utils.file_tools import io_utils
from tframe.utils.string_tools import get_time_string

from tframe.core.decorators import with_graph


class Agent(Nomear):
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
    context.note = self._note

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
  def root_path(self):
    if hub.job_dir == './': return hub.record_dir
    else: return hub.job_dir

  @property
  def note_dir(self):
    return check_path(self.root_path, hub.note_folder_name,
                      self._model.mark, create_path=hub.export_note)
  @property
  def log_dir(self):
    return check_path(self.root_path, hub.log_folder_name,
                      self._model.mark, create_path=hub.summary)
  @property
  def ckpt_dir(self):
    if hub.specified_ckpt_path is not None: return hub.specified_ckpt_path
    return check_path(self.root_path, hub.ckpt_folder_name,
                      self._model.mark, create_path=hub.save_model)
  @property
  def snapshot_dir(self):
    return check_path(self.root_path, hub.snapshot_folder_name,
                      self._model.mark, create_path=hub.snapshot)
  @property
  def model_path(self):
    """This property will be used only when checkpoint is to be saved.
        Old name format: XXXX.model
        New name example: recurrent.predictor(26.799_epochs)-train-1800
        Where XXXX denotes self._model.model_name
    """
    name = '{}.{}'.format(self._model.affix, self._model.model_name.lower())
    return os.path.join(self.ckpt_dir, name)

  @property
  def gather_path(self):
    return os.path.join(check_path(self.root_path), hub.gather_file_name)

  @property
  def gather_summ_path(self):
    return os.path.join(check_path(self.root_path), hub.gather_summ_name)

  # endregion : Paths

  # endregion : Properties

  # region : Public Methods

  def get_status_feed_dict(self, is_training):
    assert isinstance(is_training, bool)
    feed_dict = {self._is_training: is_training}
    return feed_dict

  def load(self, first_time=False):
    # TODO: when save_model option is turned off and the user want to
    #   try loading the exist model, set overwrite to False
    if not hub.save_model and hub.overwrite: return False, 0, None

    ckpt_dir = self.ckpt_dir
    if first_time and hub.mark_to_load not in ('', None):
      ckpt_dir = check_path(self.root_path, hub.ckpt_folder_name,
                            hub.mark_to_load, create_path=False)

    return load_checkpoint(ckpt_dir, self.session, self._saver)

  def save_model(self, rounds=None, suffix=None):
    """rounds is used only by trainer"""
    path = self.model_path
    if rounds is not None: path += '({:.3f}_rounds)'.format(rounds)
    if suffix is not None: path += '-{}'.format(suffix)
    save_checkpoint(path, self.session, self._saver, self._model.counter)

  def save_config_sheet(self):
    # Get tgt path
    dst_fn = f'{self._model.mark}.py'
    dst_path = os.path.join(self.ckpt_dir, dst_fn)
    if os.path.exists(dst_path): return

    # Get src path
    import shutil
    from tframe.utils.misc import date_string
    src_path = sys.argv[0]
    shutil.copy(src_path, dst_path)

    # Edit dst file
    with open(dst_path, 'r+') as f:
      content = f.readlines()
      f.seek(0)
      f.truncate()
      # Write
      f.write('import sys\n')
      def append_to_sys_argv(cmd: str):
        f.write(f"sys.argv.append('{cmd}')\n")
      for cmd in sys.argv[1:]:
        if '--train' in cmd: continue
        append_to_sys_argv(cmd)
      append_to_sys_argv('--train=False')
      append_to_sys_argv('--overwrite=False')

      # Set date to prefix if necessary
      prefix_setters = ['  th.prefix = \'{}_\'.format(date_string())\n',
                        '  th.set_date_as_prefix()\n']
      if any([ps in content for ps in prefix_setters]):
        append_to_sys_argv(f'--prefix={date_string()}_')

      f.write('# ' + '-' * 25 + ' auto generated by tframe ' + '-' * 25 + '\n')
      f.write('\n')
      # Write back original content
      f.writelines(content)

  @with_graph
  def reset_saver(self):
    """This method will be used in some very special cased, e.g. for
       saving train_stats used in dynamic evaluation (krause, 2018)
    """
    vars = self._model.variable_to_save
    self._saver = tf.train.Saver(var_list=vars, max_to_keep=2)

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
    if hub.save_model: self.save_config_sheet()

    # Launch session on self.graph
    console.show_status('Launching session ...')
    config = tf.ConfigProto()
    if hub.visible_gpu_id is not None:
      gpu_id = hub.visible_gpu_id
      if isinstance(gpu_id, int): gpu_id = '{}'.format(gpu_id)
      elif not isinstance(gpu_id, str): raise TypeError(
        '!! Visible GPU id provided must be an integer or a string')
      os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    if not hub.allow_growth:
      value = hub.gpu_memory_fraction
      config.gpu_options.per_process_gpu_memory_fraction = value
    self._session = tf.Session(graph=self._graph, config=config)
    console.show_status('Session launched')
    # Prepare some tools
    self.reset_saver()
    if hub.summary or hub.hp_tuning:
      self._summary_writer = tf.summary.FileWriter(self.log_dir)

    # Initialize all variables
    self._session.run(tf.global_variables_initializer())
    # Set init_val for pruner if necessary
    # .. if existed model is loaded, variables will be overwritten
    if hub.prune_on: context.pruner.set_init_val_lottery18()

    # Try to load exist model
    load_flag, self._model.counter, self._model.rounds = self.load(
      first_time=True)
    # Sanity check
    if hub.prune_on and hub.pruning_iterations > 0:
      if not load_flag: raise AssertionError(
        '!! Model {} should be initialized'.format(self._model.mark))

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
    elif hub.branch_suffix not in [None, '']:
      hub.mark += hub.branch_suffix
      self._model.mark = hub.mark
      console.show_status('Checkpoint switched to branch `{}`'.format(hub.mark))

    self._model.launched = True
    self.take_notes('Model launched')

    # Force re-initialize all weights in pruner (to examine rewind op)
    if hub.force_initialize:
      self._session.run([w.initializer for w in context.pruner.weights_list])
      console.show_status(
        'All weights registered to Pruner has been re-initialized.')

    # Handle structure detail here
    self._model.handle_structure_detail()

    return load_flag

  def shutdown(self):
    if hub.summary or hub.hp_tuning:
      self._summary_writer.close()
    # Clear context
    context.lr_coef = None
    context.lr_global_step = None
    context.lr_decay_steps = None
    # Clear graph
    tf.keras.backend.clear_session()
    # Close session
    self.session.close()

  def write_summary(self, summary, step=None):
    if not hub.summary: return
    if step is None:
      assert context.trainer is not None
      if hub.epoch_as_step and context.trainer.total_rounds is not None:
        step = int(context.trainer.total_rounds * 1000)
      else: step = self._model.counter
    assert isinstance(self._summary_writer, tf.summary.FileWriter)
    self._summary_writer.add_summary(summary, step)

  def save_plot(self, fig, filename):
    imtool.save_plt(fig, '{}/{}'.format(self.snapshot_dir, filename))

  # endregion : Public Methods

  # region : Public Methods for Note

  # region : For TensorViewer

  def take_down_scalars_and_tensors(self, scalars, tensors):
    assert isinstance(scalars, dict) and isinstance(tensors, dict)
    if hub.epoch_as_step and context.trainer.total_rounds is not None:
      step = int(context.trainer.total_rounds * 1000)
    else: step = self._model.counter
    self._note.take_down_scalars_and_tensors(step, scalars, tensors)

  # endregion : For TensorViewer

  # region : For SummaryViewer

  def add_to_note_misc(self, key, val):
    self._note.misc[key] = val

  def put_down_configs(self, th):
    assert isinstance(th, Config)
    self._note.put_down_configs(th.key_options)

  def put_down_criterion(self, name, value):
    self._note.put_down_criterion(name, value)

  def gather_to_summary(self):
    import pickle
    # Try to load note list into summaries
    file_path = self.gather_summ_path
    if os.path.exists(file_path):
      # with open(file_path, 'rb') as f: summary = pickle.load(f)
      summary = io_utils.load(file_path)
      assert len(summary) > 0
    else: summary = []
    # Add note to list and save
    note = self._note.tensor_free if hub.gather_only_scalars else self._note
    summary.append(note)
    io_utils.save(summary, file_path)
    # with open(file_path, 'wb') as f:
    #   pickle.dump(summary, f, pickle.HIGHEST_PROTOCOL)

    # Show status
    console.show_status('Note added to summaries ({} => {}) at `{}`'.format(
      len(summary) - 1, len(summary), file_path))

  # endregion : For SummaryViewer

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

  def show_notes(self):
    console.section('Notes')
    console.write_line(self._note.content)

  def export_notes(self, filename='notes'):
    assert hub.export_note
    # Export .txt file
    file_path = '{}/{}.txt'.format(self.note_dir, filename)
    writer = open(file_path, 'a')
    writer.write('=' * 79 + '\n')
    writer.write(self._note.content + '\n')
    writer.close()
    # Export .note file
    file_path = '{}/{}.note'.format(self.note_dir, filename)
    self._note.save(file_path)
    console.show_status('Note exported to `{}`'.format(file_path))

  def gather_notes(self, take_down_time=False):
    assert hub.gather_note
    # If gather file does not exist, create one
    with open(self.gather_path, 'a'): pass
    # Gather notes to .txt file
    line = self._note.content
    with open(self.gather_path, 'r+') as f:
      content = f.readlines()
      f.seek(0)
      f.truncate()
      if take_down_time: line = '[{}] {}'.format(get_time_string(), line)
      f.write(line + '\n')
      f.write('-' * 79 + '\n')
      f.writelines(content)
      # TODO: find a way to update immediately after training is over
    # Gather notes to .summ file
    self.gather_to_summary()

  # endregion : Public Methods for Note

  # region : Private Methods

  def _init_graph(self, graph):
    if graph is not None:
      assert isinstance(graph, tf.Graph)
      self._graph = graph
    else:
      self._graph = tf.get_default_graph()
      # self._graph = tf.Graph()

    # Initialize graph variables
    with self.graph.as_default():
      self._is_training = tf.placeholder(
        dtype=tf.bool, name=pedia.is_training)
      tf.add_to_collection(pedia.is_training, self._is_training)

      if tfr.hub.use_batch_mask:
        batch_mask = tf.placeholder(dtype=tf.bool, shape=[None],
                                    name=pedia.batch_mask)
        hub.put_into_pocket(pedia.batch_mask, batch_mask)
        tf.add_to_collection(pedia.default_feed_dict, batch_mask)

    # TODO
    # When linking batch-norm layer (and dropout layer),
    #   this placeholder will be got from default graph
    # self._graph.is_training = self._is_training
    # assert context.current_graph is not None
    if not hub.suppress_current_graph:
      context._current_graph = self._graph
    # tfr.current_graph = self._graph

  def _check_bash(self):
    command = 'tensorboard --logdir=./logs/ --port={}'.format(hub.tb_port)
    file_path = check_path(self.root_path, create_path=True)
    file_names = ['win_launch_tensorboard.bat', 'unix_launch_tensorboard.sh']
    for file_name in file_names:
      path = os.path.join(file_path, file_name)
      if not os.path.exists(path):
        f = open(path, 'w')
        f.write(command)
        f.close()

  # endregion : Private Methods

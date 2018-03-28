from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.models.feedforward import Feedforward

from tframe import console
from tframe import losses
from tframe import pedia
from tframe import metrics
from tframe import TFData

from tframe import FLAGS
from tframe import with_graph


class Predictor(Feedforward):
  def __init__(self, mark=None):
    Feedforward.__init__(self, mark)
    self._targets = None

  @property
  def description(self):
    return self.structure_string()

  @property
  def metric_is_accuracy(self):
    return pedia.memo[pedia.metric_name] == pedia.Accuracy

  @with_graph
  def build(self, loss='cross_entropy', optimizer=None,
             metric=None, metric_name='Metric'):
    Feedforward.build(self)
    # Summary placeholder
    default_summaries = []
    print_summaries = []
    # Initiate targets and add it to collection
    self._targets = tf.placeholder(self.outputs.dtype, self.outputs.get_shape(),
                                   name='targets')
    tf.add_to_collection(pedia.default_feed_dict, self._targets)

    # Define loss
    loss_function = losses.get(loss)
    with tf.name_scope('Loss'):
      self._loss = loss_function(self._targets, self.outputs)
      # TODO: with or without regularization loss?
      default_summaries.append(tf.summary.scalar('loss_sum', self._loss))
      # Try to add regularization loss
      reg_loss = self.regularization_loss
      self._loss = self._loss if reg_loss is None else self._loss + reg_loss

    # Define metric
    metric_function = metrics.get(metric)
    if metric_function is not None:
      pedia.memo[pedia.metric_name] = metric_name
      with tf.name_scope('Metric'):
        self._metric = metric_function(self._targets, self.outputs)
        print_summaries.append(tf.summary.scalar('metric_sum', self._metric))

    # Merge summaries
    self._merged_summary = tf.summary.merge(default_summaries)
    if len(print_summaries) > 0:
      self._print_summary = tf.summary.merge(print_summaries)

    # Define train step
    self._define_train_step(optimizer)

    # Print status and model structure
    self.show_building_info(FeedforwardNet=self)

    # Launch session
    self.launch_model(FLAGS.overwrite and FLAGS.train)

    # Set built flag
    self._built = True

  # TODO: does it have any difference with the one in model ?
  def _print_progress(self, epc, start_time, info_dict, **kwargs):
    # Generate loss string
    loss_strings = ['{} = {:.3f}'.format(k, info_dict[k])
                    for k in info_dict.keys()]
    loss_string = ', '.join(loss_strings)

    total_epoch = self._counter / self._training_set.batches_per_epoch
    if FLAGS.progress_bar: console.clear_line()
    console.show_status(
      'Epoch {} [{:.1f} Total] {}'.format(epc + 1, total_epoch, loss_string))
    if FLAGS.progress_bar:
      console.print_progress(progress=self._training_set.progress,
                             start_time=start_time)


  def predict(self, data):
    # Sanity check
    if not isinstance(data, TFData):
      raise TypeError('!! Input data must be an instance of TFData')
    if not self.built: raise ValueError('!! Model not built yet')
    if self._session is None:
      self.launch_model(overwrite=False)

    if data.targets is None:
      outputs = self._session.run(
        self.outputs,
        feed_dict=self._get_default_feed_dict(data, is_training=False))
      return outputs
    else:
      outputs, loss = self._session.run(
        [self.outputs, self._loss],
        feed_dict=self._get_default_feed_dict(data, is_training=False))
      return outputs, loss









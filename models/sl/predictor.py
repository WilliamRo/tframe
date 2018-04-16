from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tframe.models.feedforward import Feedforward
from tframe.models.recurrent import Recurrent

from tframe import console
from tframe import losses
from tframe import pedia
from tframe import metrics
from tframe import TFData

from tframe import config
from tframe import with_graph


class Predictor(Feedforward, Recurrent):
  """A feedforward or a recurrent predictor"""
  model_name = 'Predictor'

  def __init__(self, mark=None, net_type=Feedforward):
    """
    Construct a Predictor
    :param mark: model mark
    :param net_type: \in {Feedforward, Recurrent}
    """
    if not net_type in (Feedforward, Recurrent):
      raise TypeError('!! Unknown net type')
    self.master = net_type
    # Call parent's constructor
    net_type.__init__(self, mark)
    # Attributes
    self._targets = None

  # region : Properties

  @property
  def description(self):
    # Call Net's method
    return self.structure_string()

  @property
  def metric_is_accuracy(self):
    return pedia.memo[pedia.metric_name] == pedia.Accuracy

  # endregion : Properties

  # region : Build

  @with_graph
  def build(self, loss='cross_entropy', optimizer=None,
            metric=None, metric_name='Metric'):
    # Call parent's build method
    self.master.build(self)

    # Summary placeholder
    default_summaries = []
    # print summaries are produced during the execution of print function,
    # .. i.e. the metric summary
    print_summaries = []

    # Initiate targets and add it to collection
    self._targets = tf.placeholder(
      self._outputs.dtype, self._outputs.get_shape(), name='targets')
    tf.add_to_collection(pedia.default_feed_dict, self._targets)

    # Define loss
    loss_function = losses.get(loss)
    with tf.name_scope('Loss'):
      self._loss = loss_function(self._targets, self._outputs)
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
        self._metric = metric_function(self._targets, self._outputs)
        print_summaries.append(tf.summary.scalar('metric_sum', self._metric))

    # Merge summaries
    self._merged_summary = tf.summary.merge(default_summaries)
    if len(print_summaries) > 0:
      self._print_summary = tf.summary.merge(print_summaries)

    # Define train step
    self._define_train_step(optimizer)

    # Print status and model structure
    kwargs = {'{}'.format(self.master.__name__): self}
    self.show_building_info(**kwargs)

    # Launch session
    self.launch_model(config.overwrite)

    # Set built flag
    self._built = True

  # endregion : Build

  # region : Train

  # TODO: does it have any difference with the one in model ?
  def _print_progress(self, epc, start_time, info_dict, **kwargs):
    # Generate loss string
    loss_strings = ['{} = {:.3f}'.format(k, info_dict[k])
                    for k in info_dict.keys()]
    loss_string = ', '.join(loss_strings)

    total_epoch = self._counter / self._training_set.batches_per_epoch
    if config.progress_bar: console.clear_line()
    console.show_status(
      'Epoch {} [{:.1f} Total] {}'.format(epc + 1, total_epoch, loss_string))
    if config.progress_bar:
      console.print_progress(progress=self._training_set.progress,
                             start_time=start_time)

  # endregion : Train

  # region : Public Methods

  def predict(self, data, only_output=False, **kwargs):
    # Sanity check
    if not isinstance(data, TFData):
      raise TypeError('!! Input data must be an instance of TFData')
    if not self.built: raise ValueError('!! Model not built yet')
    if self._session is None:
      self.launch_model(overwrite=False)

    only_output = only_output or data.targets is None
    fetches = [self._outputs] + [] if only_output else [self._loss]
    feed_dict = self._get_default_feed_dict(data, is_training=False)
    return self._session.run(fetches, feed_dict=feed_dict)

  # endregion : Public Methods








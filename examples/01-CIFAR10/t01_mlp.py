import cf10_core as core
import cf10_mu as m
import tensorflow as tf
from tframe import console
from tframe.utils.misc import date_string


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'mlp'
id = 0
def model(th):
  assert isinstance(th, m.Config)
  model = m.get_container(th, flatten=True)

  for dim in th.fc_dims:
    model.add(m.Linear(dim))
    if th.use_batchnorm: model.add(m.BatchNormalization())
    model.add(m.Activation(th.spatial_activation))
    if th.dropout > 0: model.add(m.Dropout(train_keep_prob=1. - th.dropout))

  return m.finalize(th, model)


def main(_):
  console.start('{} on CIFAR-10 task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.visible_gpu_id = 1

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.spatial_activation = 'relu'
  th.developer_code = '1024-512'
  th.fc_dims = [int(s) for s in th.developer_code.split('-')]

  th.use_batchnorm = False
  th.dropout = 0.0
  th.centralize_data = True

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1
  th.batch_size = 64
  th.validation_per_round = 5

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.001

  th.patience = 5

  th.lives = 1
  th.lr_decay = 0.6

  th.clip_threshold = 10.0
  th.reset_optimizer_after_resurrection = False
  th.summary = True

  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.save_model = True
  th.overwrite = True

  th.print_cycle = 20

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = ''
  th.mark = '{}({})'.format(model_name, '-'.join(
    [str(dim) for dim in th.fc_dims])) + tail
  th.gather_summ_name = th.prefix + summ_name + tail +  '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()


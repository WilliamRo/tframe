import mn_core as core
import mn_mu as m
import tensorflow as tf
from tframe import console
from tframe.utils.misc import date_string


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'mlp'
id = 1
def model(th):
  assert isinstance(th, m.Config)
  model = m.get_container(th, flatten=True)
  # Add hidden layers
  for i, n in enumerate([int(s) for s in th.archi_string.split('-')]):
    model.add(m.Dense(n))
    if i > 0 and th.use_batchnorm: model.add(m.BatchNormalization())
    model.add(m.Activation(th.spatial_activation))
  # Finalize model
  return m.finalize(th, model)


def main(_):
  console.start('{} on MNIST task'.format(model_name.upper()))

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
  th.suffix = '_t00'
  th.visible_gpu_id = 0

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.spatial_activation = 'relu'
  th.use_batchnorm = True
  th.archi_string = '200-100'

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 128
  th.print_cycle = 20
  th.validation_per_round = 2

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.003

  th.patience = 2
  th.early_stop = True
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.save_model = True
  th.overwrite = True

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + th.suffix +  '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

import cf10_core as core
import cf10_mu as m
import tensorflow as tf
from tframe import console
from tframe.utils.misc import date_string


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'residual'
id = 2
def model(th):
  assert isinstance(th, m.Config)
  model = m.get_container(th, flatten=False)

  model.add(m.Conv2D(64, kernel_size=3, strides=1))
  model.add(m.Activation('relu'))
  model.add(m.MaxPool2D(2, strides=2))

  # Branched begin
  res_block = model.add(inter_type=model.CONCAT)

  main_branch = res_block.add()
  main_branch.add(m.Conv2D(192, kernel_size=3, strides=1))
  main_branch.add(m.Activation('relu'))
  main_branch.add(m.MaxPool2D(2, strides=2))

  shortcut = res_block.add()
  shortcut.add(m.Conv2D(20, kernel_size=3, strides=2))

  # Branches end

  for filters in (384, 256, 256):
    model.add(m.Conv2D(filters, kernel_size=3))
    model.add(m.Activation('relu'))

  model.add(m.MaxPool2D(3, strides=2))
  model.add(m.Flatten())

  for dim in (2048, 2048):
    if th.dropout > 0: model.add(m.Dropout(1. - th.dropout))
    model.add(m.Linear(dim))
    model.add(m.Activation('relu'))

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
  prefix = '{}_'.format(date_string())
  suffix = ''
  th.visible_gpu_id = 0

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.dropout = 0.2

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 64
  th.validation_per_round = 5

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.001

  th.patience = 5

  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.save_model = True
  th.overwrite = True

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = suffix
  th.mark = prefix + '{}({}){}'.format(model_name, th.num_layers, tail)
  th.gather_summ_name = prefix + summ_name + tail +  '.sum'
  core.activate(True)


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()


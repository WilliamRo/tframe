import tensorflow as tf
import core
from tframe import console
import model_lib as models


def main(_):
  console.start('Example task')

  # Configurations
  th = core.th
  th.model = models.example_model
  # ...

  th.epoch = 10
  th.learning_rate = 1e-4
  th.batch_size = 64
  th.validation_per_round = 10
  th.print_cycle = 100

  # th.train = False
  # th.smart_train = True
  th.max_bad_apples = 4
  th.lr_decay = 0.6

  th.save_model = False
  th.overwrite = True
  th.export_note = True
  th.summary = False
  th.monitor = False

  description = '0'
  th.mark = 'example_model_{}'.format(description)

  core.activate()


if __name__ == '__main__':
  tf.app.run()




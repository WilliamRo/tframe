import tensorflow as tf
import core
from tframe import console
import model_lib as models


def main(_):
  console.start('MLP task')

  # Configurations
  th = core.th
  th.model = models.mlp
  th.num_blocks = 2
  th.memory_depth = 80
  multiplier = 4
  th.hidden_dim = th.memory_depth * multiplier
  th.actype1 = 'relu'

  th.epoch = 50000
  th.learning_rate = 1e-4
  th.batch_size = 64
  th.validation_per_round = 5
  th.print_cycle = 50
  th.shuffle = True

  # th.train = False
  th.smart_train = True
  th.max_bad_apples = 4
  th.lr_decay = 0.5

  th.save_model = True
  th.overwrite = True
  th.export_note = True
  th.summary = True
  th.monitor = False

  description = ''
  th.mark = '{}x[{}x{}]{}'.format(
    th.num_blocks, th.memory_depth, multiplier, description)

  core.activate()


if __name__ == '__main__':
  tf.app.run()




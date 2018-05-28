import tensorflow as tf
import core
from tframe import console
import model_lib as models


def main(_):
  console.start('Example task')

  # Configurations
  th = core.th
  th.model = models.mlp
  th.num_blocks = 2
  th.hidden_dim = 100
  th.actype1 = 'relu'

  th.epoch = 50
  th.learning_rate = 1e-4
  th.batch_size = 64
  th.validation_per_round = 2
  th.print_cycle = 20
  th.shuffle = True

  # th.train = False
  th.smart_train = True
  th.max_bad_apples = 4
  th.lr_decay = 0.6

  th.save_model = True
  th.overwrite = True
  th.export_note = True
  th.summary = True
  th.monitor = False

  description = ''
  th.mark = 'mlp_{}x{}{}'.format(th.hidden_dim, th.num_blocks, description)

  export_false = True
  core.activate(export_false=export_false)


if __name__ == '__main__':
  tf.app.run()




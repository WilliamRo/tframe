import tensorflow as tf
import core
from tframe import console
from tframe.utils.misc import mark_str as ms
import model_lib as models


def main(_):
  console.start('MLP task')

  # Configurations
  th = core.th
  th.model = models.mlp
  th.fc_dims = [800, 500]
  th.actype1 = 'relu'

  th.epoch = 50
  th.learning_rate = 1e-5
  th.batch_size = 64
  th.validation_per_round = 2
  th.print_cycle = 20
  th.shuffle = True

  th.train = False
  th.smart_train = False
  th.max_bad_apples = 4
  th.lr_decay = 0.6

  th.save_model = True
  th.overwrite = True
  th.export_note = True
  th.summary = True
  th.monitor = False

  description = ''
  th.mark = 'mlp_{}{}'.format(ms(th.fc_dims), description)

  export_false = True
  core.activate(export_false=export_false)


if __name__ == '__main__':
  tf.app.run()




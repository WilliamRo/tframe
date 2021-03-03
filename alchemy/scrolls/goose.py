from .scroll_base import Scroll


class Goose(Scroll):

  name = 'Goose'

  def get_next_hyper_parameter(self):
    return None
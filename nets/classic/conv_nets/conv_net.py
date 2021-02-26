from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe.nets.net import Net


class ConvNet(object):

  def add_to(self, model):
    assert isinstance(model, Net)
    for f in self._get_layers(): model.add(f)


  def _get_layers(self):
    raise NotImplemented


  @staticmethod
  def parse_archi_string(archi_string: str):
    return [[int(s) for s in ss.split('-')]
            for ss in archi_string.split('=')]


if __name__ == '__main__':
  for s in ConvNet.parse_conv_fc('6-16=120-84'): print(s)
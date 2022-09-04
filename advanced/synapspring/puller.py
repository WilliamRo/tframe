from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
from tframe import context
from tframe.core.nomear import Nomear
from tframe import hub as th
from tframe import console

import numpy as np

from .springs.spring_base import SpringBase



class Puller(Nomear):

  def __init__(self, model):
    assert th.cl_reg_on

    self._model = model
    self.shadows = OrderedDict()

    self.spring: SpringBase = self._init_spring()
    self._safely_register_customized_loss_f_net()

  # region: Public Methods

  # endregion: Public Methods

  # region: Private Methods

  def _init_spring(self) -> SpringBase:
    key = th.cl_reg_config.split(':')[0]
    if key in ('synaptic-intelligence', 'si'):
      from .springs.si import SynapticIntelligence as Spring
    elif key in ('EWC', 'ewc'):
      pass
    elif key in ('l2', ):
      from .springs.spring_base import SpringBase as Spring
    elif key in context.depot:
      Spring = context.depot[key]
    else: raise KeyError(f'!! Unknown spring type `{key}')

    console.show_status(f'Spring type `{key}` registered to puller.',
                        symbol=f'[Puller]')

    assert issubclass(Spring, SpringBase)
    return Spring(self._model)

  def _safely_register_customized_loss_f_net(self):
    f_net = self._calculate_spring_loss
    if callable(context.customized_loss_f_net):
      def f_net(model):
        loss_list: list = context.customized_loss_f_net(model)
        loss_list.extend(self._calculate_spring_loss(model))
        return loss_list
    context.customized_loss_f_net = f_net

  def _calculate_spring_loss(self, model) -> list:
    return [self.spring.calculate_loss()]

  # endregion: Private Methods

from tframe.core.nomear import Nomear
from tframe import tf
from tframe import hub as th



class SpringBase(Nomear):

  def __init__(self):
    # Every spring use shadow vars
    th.create_shadow_vars = True


  def calculate_loss(self, model) -> tf.Tensor:
    from tframe import Predictor
    assert isinstance(model, Predictor)

    vars = model.var_list
    shadows = model.shadows
    assert len(vars) == len(shadows)

    loss_list = []
    for v in vars:
      s = shadows[v]
      loss_list.append(tf.reduce_mean(tf.square(s - v)))

    return tf.multiply(th.cl_reg_lambda, tf.add_n(loss_list), name='cl_reg_l2')


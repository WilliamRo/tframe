from tframe import console
from tframe import hub as th
from tframe import mu



def get_model_for_test():
  model = mu.Predictor(mark='99-02-02-01', net_type=mu.Recurrent)
  model.add(mu.Input(sample_shape=[16]))
  model.add(mu.LSTM(128))
  model.add(mu.Dense(16))
  model.build(loss='mse')
  return model



if __name__ == '__main__':
  console.suppress_logging()

  th.save_model = False

  get_model_for_test().rehearse(
    export_graph=False, build_model=False, mark='99-02-02-01')

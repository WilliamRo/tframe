import numpy as np

from tframe.utils.tensor_viewer.plugin import Plugin


def _w2sig(w):
  x = np.reshape(w, [-1])
  square = x * x
  a = square / (np.sum(square) + 1e-6)

  N = a.size
  U = np.zeros([N, N], dtype=np.float32)
  Y, X = np.meshgrid(range(N), range(N))
  U[X <= Y] = 1.0

  return U @ a


def _recursive_modify(v_dict, level=0):
  if len(v_dict) == 0: return
  assert isinstance(v_dict, dict)
  if isinstance(list(v_dict.values())[0], dict):
    for e_key, e_dict in v_dict.items():
      print('>> Modifying dict {} ...'.format(e_key))
      _recursive_modify(e_dict, level=level + 1)
    return
  # Here the values in v_dict must be lists
  for key in list(v_dict.keys()):
    if key not in ['Significance']: continue
    v_dict[key] = [_w2sig(v) for v in v_dict[key]]


def modifier(v_dict):
  print('>> Modifying by sig_curve ...')
  _recursive_modify(v_dict)


plugin = Plugin(dict_modifier=modifier)

import numpy as np


def one_hot(num_classes, hot_index):
  probs = np.zeros(shape=(num_classes,))
  probs[hot_index] = 1.0
  return probs


def q_prob(num_classes, max_prob, max_index=0, avg_rest=True):
  # Sanity check
  assert isinstance(num_classes, int) and num_classes > 0
  assert isinstance(max_index, int) and 0 <= max_index < num_classes
  assert 0 <= max_prob <= 1

  probs = np.zeros(shape=(num_classes,))
  probs[max_index] = max_prob

  rest_indices = list(range(num_classes))
  rest_indices.pop(max_index)
  rest_indices = np.array(rest_indices)

  rest_probs = (np.ones(shape=(num_classes - 1,)) if avg_rest
                else np.random.rand(num_classes - 1))
  rest_probs = (1 - max_prob) * rest_probs / np.sum(rest_probs)

  probs[rest_indices] = rest_probs
  assert np.abs(sum(probs) - 1.0) < 1e-7
  return probs


def cross_entropy(p, q):
  """H(p, q) = E_p[-log(q)] = H(p) + D_KL(p || q)"""
  assert isinstance(p, np.ndarray) and isinstance(q, np.ndarray)
  assert p.shape == q.shape and len(q.shape) == len(q.shape) == 1
  return np.sum(p * (-np.log(q)))



if __name__ == '__main__':
  num_classes = 41
  max_probability = [1 / num_classes, 2 / num_classes, 3 / num_classes]

  for i, prob in enumerate(max_probability):
    p = one_hot(num_classes, hot_index=0)
    q = q_prob(num_classes, prob, avg_rest=False)
    print('[{}] max_prob = {:.4f} => cross_entropy = {:.4f}'.format(
      i, prob, cross_entropy(p, q)))


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def apportion(length_list, batch_size):
  """This method is first created to deal with the allocate problem
     in customized gen_rnn_batches method of FI-2010 data set.
  :param length_list: a list containing lengths. e.g. structure of a SequenceSet
  :param batch_size: an non-negative integer
  :return: a list of batches for each sequence, sum of which equals to
           the specified batch_size.
  """
  L = int(sum(length_list) / batch_size)
  if L > min(length_list):
    raise ValueError('L ({}) must be less than minimum sequence '
                     'length ({})'.format(L, min(length_list)))
  batches_dec = [length / L for length in length_list]
  batches = [int(np.round(b)) for b in batches_dec]
  extra_dec = [b - bd for b, bd in zip(batches, batches_dec)]
  extra_rank = np.argsort(extra_dec)

  lack = batch_size - sum(batches)
  assert lack < len(length_list)
  if lack > 0:
    # If sub-sequences are not enough, fill batches with ones with least
    #   extra_dec
    for i in extra_rank[:lack]: batches[i] += 1
  elif lack < 0:
    # If the total number of sub-sequences to sample is more than batch_size,
    #   remove those with least lack values
    for i in extra_rank[lack:]: batches[i] -= 1

  # Check and return
  assert sum(batches) == batch_size and len(batches) == len(length_list)
  return batches


def spread(length, N, L, radius=0):
  """Generate the start indices of N sub-sequences of length L sampled from a
     sequence of given `length`.
  """
  bound = lambda i: min(max(0, i), length - L)
  # Divide the sequence
  SL = int(length / N)
  section_indices = [[i * SL, (i + 1) * SL] for i in range(N)]
  section_indices[-1][-1] = length
  start_indices = []
  for start_i, end_i in section_indices:
    sl = end_i - start_i
    if L <= sl: index = np.random.randint(start_i, end_i - L + 1)
    else: index = np.random.randint(end_i - L, start_i + 1)
    index = bound(index)
    if radius > 0: index = bound(index + np.random.randint(-radius, radius))
    start_indices.append(index)

  # Sanity check and return
  assert len(start_indices) == N
  return start_indices


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from tframe import console


class NoteList(object):
  def __init__(self, sum_path):
    assert isinstance(sum_path, str)
    self.notes = None
    self.summary_path = None
    self.load(sum_path)

  def load(self, path):
    try:
      with open(path, 'rb') as f:
        self.notes = pickle.load(f)
      assert isinstance(self.notes, list)
      self.summary_path = path
    except:
      print('!! Failed to load {}'.format(path))

  def save(self):
    with open(self.summary_path, 'wb') as f:
      pickle.dump(self.notes , f, pickle.HIGHEST_PROTOCOL)
    console.show_status('Note list (length {}) saved to `{}`'.format(
      len(self.notes), self.summary_path))


if __name__ == '__main__':
  path = r''
  nl = NoteList(path)
  key = 'total_params'
  for i, note in enumerate(nl.notes):
    if key in note._configs.keys():
      note.configs[key] = int(note.configs[key])
      console.show_status('Note[{}] fixed.'.format(i))
  nl.save()
  _ = None

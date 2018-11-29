from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import pickle
from tframe.utils.note import Note


class Context(object):
  """The context of a NoteViewer. Usually stores the note file info and
     the instance of a note"""
  PRESET_INACTIVE_CRI = ('Total Rounds', 'Mean Record')

  def __init__(self, default_inactive_flags=(), default_inactive_criteria=()):
    self.summary_file_path = None
    self.notes = []

    self.active_flag_set = set()
    self.inactive_flag_set = set()
    self.default_inactive_flags = default_inactive_flags
    self.flag_value_dict = dict()

    self.active_criteria_set = set()
    self.inactive_criteria_set = set()
    self.default_inactive_criteria = default_inactive_criteria
    self.default_inactive_criteria += self.PRESET_INACTIVE_CRI

  # region : Properties

  @property
  def summary_file_name(self):
    last_level = 2
    if self.summary_file_path is None: return None
    assert isinstance(self.summary_file_path, str)
    return '/'.join(re.split(r'/|\\', self.summary_file_path)[-last_level:])

  # endregion : Properties

  # region : Public Methods

  def set_notes_by_path(self, summ_file_path):
    # Sanity check
    assert isinstance(summ_file_path, str)
    # Try to load note file
    try:
      with open(summ_file_path, 'rb') as f:
        self.notes = pickle.load(f)
      assert isinstance(self.notes, list)
      self.summary_file_path = summ_file_path
    except:
      print('!! Failed to load {}'.format(summ_file_path))
      return
    # Print status
    print('>> Loaded notes from `{}`'.format(self.summary_file_path))

    # Initialize flags and criteria
    self._init_flags()
    self._init_criteria()

  # endregion : Public Methods

  # region : Private Methods

  def _get_intersection_and_union(self, dict_attr):
    intersection, union = set(), set()
    for i, note in enumerate(self.notes):
      # Get the specific dictionary attribute of note
      attr = getattr(note, dict_attr)
      assert isinstance(attr, dict)
      keys = attr.keys()
      # Update intersection and union
      if i == 0:
        intersection, union = set(keys), set(keys)
      else:
        key_set = set(attr.keys())
        intersection.intersection_update(key_set)
        union.update(key_set)

    return intersection, union

  def _init_flags(self):
    intersection, union = self._get_intersection_and_union('configs')

    # Remove default inactive flags from intersection
    self.active_flag_set = intersection - intersection.intersection(
      set(self.default_inactive_flags))
    self.inactive_flag_set = union - self.active_flag_set

    def get_flag_values(k):
      values = set()
      for note in self.notes:
        val = note.configs.get(k, None)
        if val is not None: values.add(val)
      assert len(values) > 0
      return tuple(values)

    # Get flag_values
    for key in union:
      self.flag_value_dict[key] = get_flag_values(key)

  def _init_criteria(self):
    intersection, union = self._get_intersection_and_union('criteria')

    # Remove default inactive flags from intersection
    self.active_criteria_set = union - union.intersection(
      set(self.default_inactive_criteria))
    self.inactive_criteria_set = union - self.active_criteria_set

  # endregion : Private Methods


if __name__ == '__main__':
  pass

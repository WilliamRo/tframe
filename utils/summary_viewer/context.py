from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import pickle
from collections import OrderedDict
from tframe.utils.note import Note


class Context(object):
  """The context of a NoteViewer. Usually stores the note file info and
     the instance of a note"""
  PRESET_INACTIVE_CRI = ('Total Rounds', 'Mean Record')
  PRESET_INACTIVE_CFG = ('export_tensors_to_note',)

  def __init__(self,
               default_inactive_flags=(),
               default_inactive_criteria=(),
               flags_to_ignore=()):
    self.summary_file_path = None
    self.notes = []

    self.active_flag_set = set()
    self.inactive_flag_set = set()
    self.default_inactive_flags = default_inactive_flags
    self.default_inactive_flags += self.PRESET_INACTIVE_CFG
    self.flag_value_dict = OrderedDict()

    self.active_criteria_set = set()
    self.inactive_criteria_set = set()
    self.default_inactive_criteria = default_inactive_criteria
    self.default_inactive_criteria += self.PRESET_INACTIVE_CRI

    self.flags_to_ignore = flags_to_ignore

  # region : Properties

  @property
  def summary_file_name(self):
    last_level = 2
    if self.summary_file_path is None: return None
    assert isinstance(self.summary_file_path, str)
    return '/'.join(re.split(r'/|\\', self.summary_file_path)[-last_level:])

  @property
  def active_flag_list(self):
    return self._set2list(self.active_flag_set)

  @property
  def inactive_flag_list(self):
    return self._set2list(self.inactive_flag_set)

  @property
  def active_criteria_list(self):
    return self._set2list(self.active_criteria_set)

  @property
  def inactive_criteria_list(self):
    return self._set2list(self.inactive_criteria_set)

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

  def reload(self):
    if not isinstance(self.summary_file_path, str): return
    pre_length = len(self.notes)
    try:
      with open(self.summary_file_path, 'rb') as f:
        self.notes = pickle.load(f)
      assert isinstance(self.notes, list)
    except:
      print('!! Failed to reload {}'.format(self.summary_file_path))
      return
    # Print status
    print('>> Reloaded notes from `{}`'.format(self.summary_file_path))
    delta = len(self.notes) - pre_length
    assert delta >= 0
    delta_str = 'No' if delta == 0 else '{}'.format(delta)
    print('>> {} notes added.'.format(delta_str))

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
    intersection -= set(self.flags_to_ignore)
    union -= set(self.flags_to_ignore)

    # Remove default inactive flags from intersection
    self.active_flag_set = intersection - intersection.intersection(
      set(self.default_inactive_flags))
    self.inactive_flag_set = union - self.active_flag_set

    def get_flag_values(k):
      values = set()
      for note in self.notes:
        if k in note.configs.keys():
          values.add(note.configs[k])
      assert len(values) > 0
      values = list(values)
      values.sort()
      return tuple(values)

    # Get flag_values
    sorted_union = list(union)
    sorted_union.sort()
    for key in sorted_union:
      self.flag_value_dict[key] = get_flag_values(key)

  def _init_criteria(self):
    intersection, union = self._get_intersection_and_union('criteria')

    # Remove default inactive flags from intersection
    self.active_criteria_set = union - union.intersection(
      set(self.default_inactive_criteria))
    self.inactive_criteria_set = union - self.active_criteria_set

  @staticmethod
  def _set2list(s, reverse=False):
    l = list(s)
    l.sort(reverse=reverse)
    return l

  # endregion : Private Methods


if __name__ == '__main__':
  pass

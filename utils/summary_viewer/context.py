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

  def set_notes(self, summaries):
    if isinstance(summaries, str):
      # Try to load note file
      try:
        with open(summaries, 'rb') as f:
          self.notes = pickle.load(f)
        assert isinstance(self.notes, list)
        self.summary_file_path = summaries
      except:
        import traceback
        print('!! Failed to load {}'.format(summaries))
        print('Error message:' + '\n' + '-' * 79 + 'x')
        print(traceback.format_exc() + '-' * 79 + 'x')
        return
      # Print status
      print('>> Loaded notes from `{}`'.format(self.summary_file_path))
    else:
      assert isinstance(summaries, list)
      self.notes = summaries
      self.summary_file_path = 'Unknown'
      print('>> {} notes set to viewer'.format(len(summaries)))

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
          value = note.configs[k]
          # For type value
          if isinstance(value, type):
            value = str(value)
            m = re.match(r"<class '([\w]+.)+([\w]+)'>", value)
            if m is not None: value = m.group(1)
            # Set string value back to note
            note.configs[k] = value
          # For list value
          if isinstance(value, list):
            value = tuple(value)
            note.configs[k] = value
          # TODO: for values if str(value) is too long
          if not isinstance(value, str) and len(str(value)) > 15:
            # For float value
            if isinstance(value, float):
              value = '{:.5f}'.format(value)
              note.configs[k] = value
            else:
              m = re.match(r"<class '([\w]+.)+([\w]+)'>", str(type(value)))
              if m is not None:
                value = m.group(1)
                note.configs[k] = value
          values.add(value)
      assert len(values) > 0
      values = list(values)

      # TODO: workaround for avoiding sort stuff like (None, 4)
      try: values.sort()
      except: pass

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

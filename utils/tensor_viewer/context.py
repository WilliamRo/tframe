from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tframe.utils.note import Note


class Context(object):
  """The context of a NoteViewer. Usually stores the note file info and
     the instance of a note"""
  def __init__(self):
    self.note = None
    self.note_file_path = None


  @property
  def note_file_name(self):
    if self.note_file_path is None: return None
    assert isinstance(self.note_file_path, str)
    return re.split(r'/|\\', self.note_file_path)[-1]


  def set_note_by_path(self, note_file_path):
    # Sanity check
    assert isinstance(note_file_path, str)
    # Try to load note file
    try:
      self.note = Note.load(note_file_path)
      self.note_file_path = note_file_path
    except:
      print('!! Failed to load {}'.format(note_file_path))
      return
    # Print status
    print('>> Loaded note file: {}'.format(self.note_file_name))


  def set_note(self, note=None, note_path=None):
    if note is not None:
      assert isinstance(note, Note)
      self.note = note
      self.note_file_path = None
    else:
      assert isinstance(note_path, str)
      self.set_note_by_path(note_path)

    # Show the content of the note
    print('-' * 79)
    print(self.note.content)
    print('-' * 79)


if __name__ == '__main__':
  pass

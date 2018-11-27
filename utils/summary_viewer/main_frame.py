from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  import tkinter as tk
  import tkinter.ttk as ttk

  from tframe.utils.note import Note
  from tframe.utils.viewer_base.main_frame import Viewer

  from tframe.utils.summary_viewer import key_events
  from tframe.utils.summary_viewer.context import Context
  from tframe.utils.summary_viewer.header_control import HeaderControl
  from tframe.utils.summary_viewer.config_control import ConfigPanel

except Exception as e:
  print(' ! {}'.format(e))
  print(' ! Summary Viewer is disabled, install tkinter to enable it')


class SummaryViewer(Viewer):
  """Summary Viewer for tframe summary"""
  SIZE = 400

  def __init__(self, summary_path=None, **kwargs):
    # Call parent's constructor
    Viewer.__init__(self)
    self.master.resizable(False, False)

    # Attributes
    self.context = Context(
      kwargs.get('default_inactive_flags', ()),
      kwargs.get('default_inactive_criteria', ()))

    # Layout
    self.header = HeaderControl(self)
    self.main_panel = ttk.Frame(self)
    self.config_panel = ConfigPanel(self.main_panel)
    self.criteria_panel = None

    # Create layout and bind key events
    self._set_global_style()
    self._create_layout()
    self._bind_key_events()

    # Try to load summary
    if summary_path is not None:
      assert isinstance(summary_path, str)
      self.set_notes_by_path(summary_path)
    else:
      self.global_refresh()

  # region : Public Methods

  def set_notes_by_path(self, summary_path):
    # Set context
    self.context.set_notes_by_path(summary_path)
    # Refresh
    self.global_refresh()

  def global_refresh(self):
    # Refresh title
    title = 'Summary Viewer'
    if self.context.summary_file_name is not None:
      title += ' - {}'.format(self.context.summary_file_name)
    self.form.title(title)

    # Initialize config controls
    self.config_panel.initialize_config_controls()

    # Refresh header
    self.header.refresh()

    # Do local refresh
    self.local_refresh()

  def local_refresh(self):
    # Refresh config panel
    self.config_panel.refresh()

  # endregion : Public Methods

  # region : Private Methods

  def _set_global_style(self):
    # Set white background
    set_white_bg = lambda name: self.set_style(name, background='white')

    set_white_bg(self.WidgetNames.TLabel)
    set_white_bg(self.WidgetNames.TFrame)
    set_white_bg(self.WidgetNames.TLabelframe)
    self.set_style(self.WidgetNames.TLabelframe, 'Label', background='white',
                   reverse=False)

    # ...
    # self.set_style(self.WidgetNames.TCombobox, )

  def _bind_key_events(self):
    # Bind Key Events
    self.form.bind('<Key>', lambda e: key_events.on_key_press(self, e))
    self.form.bind('<Control-o>', lambda e: key_events.load_notes(self, e))

  def _create_layout(self):
    # Header
    self.header.configure(height=50, width=600, padding=5)
    # HeaderControl.COLOR = 'green3'
    self.header.load_to_master()

    # Main panel for config panel and criterion panel
    self.main_panel.pack(fill=tk.BOTH)

    # Config panel
    self.config_panel.configure(height=200, width=400, padding=5)
    self.config_panel.load_to_master()

    # TODO:
    frame_style = lambda n, bg: self.set_style(
      self.WidgetNames.TFrame, n, background=bg)

    # Right
    right = ttk.Frame(self.main_panel, style=frame_style('right', 'dark orange'))
    right.configure(height=400, width=400)
    right.pack(fill=tk.BOTH, expand=1, side=tk.RIGHT)

    # Bottom
    # bottom = ttk.Frame(self, style=frame_style('bottom', 'tomato'))
    # bottom.configure(height=50, width=600)
    # bottom.pack(fill=tk.BOTH, expand=1, side=tk.TOP)

    # Pack self
    self.pack(fill=tk.BOTH, expand=True)

  # endregion : Private Methods


if __name__ == '__main__':
  from tframe.utils.summary_viewer import main_frame

  summ_path = r'E:\rnn_club\01-ERG\records_shem\gather.sum'
  summ_path = r'E:\rnn_club\01-ERG\records_shem\test.sum'
  summ_path = None
  viewer = main_frame.SummaryViewer(
    summary_path=summ_path,
    default_inactive_flags=(
      'patience',
      'shuffle',
      'epoch',
      'early_stop',
      'warm_up_thres',
      'mark',
      'batch_size',
      'save_mode',
    ),
    default_inactive_criteria=(
      'Mean Record',
      'Record',
    )
  )
  viewer.show()



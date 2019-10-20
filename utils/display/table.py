from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Table(object):

  def __init__(self, *widths, tab=4, margin=2, buffered=False):
    assert len(widths) > 0
    self.columns = len(widths)
    self._widths = widths
    self._margin = margin
    self._tab = tab
    self._col_fmt = ['{}'] * self.columns
    self._align = None
    self._buffered = buffered
    self._buffer = []

  @property
  def content(self): return '\n' + '\n'.join(self._buffer)

  @property
  def tab(self): return ' ' * self._tab

  @property
  def hline_width(self):
    return sum(self._widths) + self._tab * (self.columns - 1) + 2 * self._margin

  def print(self, string):
    if self._buffered: self._buffer.append(string)
    else: print(string)

  def hline(self): self.print('-' * self.hline_width)

  def hdash(self): self.print('- ' * int(self.hline_width / 2))

  def dhline(self): self.print('=' * self.hline_width)

  def specify_format(self, *fmts, align=None):
    assert len(fmts) == self.columns
    self._col_fmt = ['{}' if f in (None, '') else f for f in fmts]
    if align is not None:
      self._align = align
      assert len(align) == self.columns

  def _get_line(self, cells):
    if self._align is None: return self.tab.join(
      [(('{:>' if i > 0 else '{:<') + str(w) + '}').format(c[:w])
       for i, (c, w) in enumerate(zip(cells, self._widths))])
    else: return self.tab.join(
      [(('{:>' if a == 'r' else '{:<') + str(w) + '}').format(c[:w])
        for c, w, a in zip(cells, self._widths, self._align)])

  def _print_with_margin(self, content):
    margin = ' ' * self._margin
    self.print('{}{}{}'.format(margin, content, margin))

  def print_header(self, *header, hline=True):
    if hline: self.hline()
    self.print_row(*header)
    if hline: self.dhline()

  def print_row(self, *cells):
    assert len(cells) == self.columns
    cells = [c if isinstance(c, str) else fmt.format(c)
             for c, fmt in zip(cells, self._col_fmt)]
    self._print_with_margin(self._get_line(cells))

  def print_buffer(self, indent=0):
    for row in self._buffer: print('{}{}'.format(' ' * indent, row))


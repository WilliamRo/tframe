import pprint


class Console(object):

  def __init__(self):
    self._default_length = 70
    self._default_title = "main"
    self._symbol = ">>"
    self._pp = pprint.PrettyPrinter()

  def start(self, title=None, length=None):
    title = title or self._default_title
    length = length or self._default_length
    print("-> Start of %s\n%s" % (title, '-' * length))

  def end(self, title=None, length=None):
    title = title or self._default_title
    length = length or self._default_length
    print("%s\n|> End of %s" % (('-' * length), title))

  def section(self, contents):
    print("=" * self._default_length)
    print("::%s" %  contents)
    print("=" * self._default_length)

  def print(self, content):
    print("%s %s" % (self._symbol, content))

  def pprint(self, content):
    self._pp.pprint(content)
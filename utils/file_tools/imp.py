# ROMA

from importlib.util import module_from_spec
from importlib.util import spec_from_file_location


def import_from_path(module_path):
  """Import a module from path.
  Reference: https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

  :param module_path: module path
  :return: module
  """
  spec = spec_from_file_location(module_path, module_path)
  mod = module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod
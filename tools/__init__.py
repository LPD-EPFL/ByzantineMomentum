# coding: utf-8
###
 # @file   __init__.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2020 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Bunch of useful tools, but each too small to have its own package.
###

import io
import os
import pathlib
import sys
import threading
import traceback

# ---------------------------------------------------------------------------- #
# User exception base class, print string representation and exit(1) on uncaught

class UserException(Exception):
  """ User exception base class.
  """
  pass

# ---------------------------------------------------------------------------- #
# Context and color management

class Context:
  """ Per-thread context and color management static class.
  """

  # Constants
  __colors  = { "header": "\033[1;30m",
    "red": "\033[1;31m", "error": "\033[1;31m",
    "green": "\033[1;32m", "success": "\033[1;32m",
    "yellow": "\033[1;33m", "warning": "\033[1;33m",
    "blue": "\033[1;34m", "info": "\033[1;34m",
    "gray": "\033[1;30m", "trace": "\033[1;30m" }
  __clrend = "\033[0m"

  # Thread-local variables
  __local = threading.local()

  @classmethod
  def __local_init(self):
    """ Initialize the thread local data if necessary.
    """
    if not hasattr(self.__local, "stack"):
      self.__local.stack  = []            # List of pairs (context name, color code)
      self.__local.header = ""            # Current header string
      self.__local.color  = self.__clrend # Current color code

  @classmethod
  def __rebuild(self):
    """ Rebuild the header and apply the current color.
    """
    # Collect current header and color
    header = ""
    color  = None
    for ctx, clr in reversed(self.__local.stack):
      if ctx is not None:
        header = "[" + ctx + "] " + header
      if clr is not None:
        if color is None:
          color = clr
    if color is None:
      color = self.__clrend
    # Prepend thread name if not main thread
    cthrd = threading.current_thread()
    if cthrd != threading.main_thread():
      header = "[" + cthrd.name + "] " + header
    # Store the new header and color
    self.__local.header = header
    self.__local.color  = color

  @classmethod
  def _get(self):
    """ Get the thread-local header and color.
    Returns:
      Current header, begin header color, begin color, ending color
    """
    self.__local_init()
    return self.__local.header, self.__colors["header"], self.__local.color, self.__clrend

  def __init__(self, cntxtname, colorname):
    """ Color selection constructor.
    Args:
      cntxtname Context name (None for none)
      colorname Color name (None for no change)
    """
    # Color code resolution
    if colorname is None:
      colorcode = None
    else:
      assert colorname in type(self).__colors, "Unknown color name " + repr(colorname)
      colorcode = type(self).__colors[colorname]
    # Finalization
    self.__pair = (cntxtname, colorcode)

  def __enter__(self):
    """ Enter context.
    Returns:
      self
    """
    type(self).__local_init()
    type(self).__local.stack.append(self.__pair)
    type(self).__rebuild()
    return self

  def __exit__(self, *args, **kwargs):
    """ Leave context.
    Args:
      ... Ignored arguments
    """
    type(self).__local.stack.pop()
    type(self).__rebuild()

class ContextIOWrapper:
  """ Context-aware text IO wrapper class.
  """

  def __init__(self, output, nocolor=None):
    """ New line no color assumed constructor.
    Args:
      output  Wrapped output
      nocolor Whether to apply colors or not (if None, no color for non-TTY)
    """
    # Check whether to apply coloring if unset
    if nocolor is None:
      nocolor = not output.isatty()
    # Finalization
    self.__newline = True # At a new line
    self.__colored = True # Color has been applied
    self.__output  = output
    self.__nocolor = nocolor

  def __getattr__(self, name):
    """ Forward non-overloaded attributes.
    Args:
      name Non-overloaded attribute name
    Returns:
      Non-overloaded attribute
    """
    return getattr(self.__output, name)

  def write(self, text):
    """ Wrap the given text with the context if necessary.
    Args:
      text Text to update and write
    Returns:
      Forwarded value
    """
    # Get the current context
    header, clrheader, clrbegin, clrend = Context._get()
    if self.__nocolor:
      clrheader = ""
      clrbegin = ""
      clrend = ""
    # Prepend the header to every line
    lines = text.splitlines(True)
    text  = ""
    for line in lines:
      if self.__newline:
        text += clrheader + header
      text += clrbegin
      self.__newline = True
      text += line
    if len(lines) > 0 and lines[-1][-len(os.linesep):] != os.linesep:
      self.__newline = False
    # Write the modified text with the right color
    return self.__output.write(text + clrend)

def _make_color_print(color):
  """ Build the closure that wrap a 'print' inside a colored context.
  Args:
    color Target color name
  Returns:
    Print wrapper closure
  """
  def color_print(*args, context=None, **kwargs):
    """ Print in 'color'.
    Args:
      context Context name to use
      ...     Forwarded arguments
    Returns:
      Forwarded return value
    """
    with Context(context, color):
      return print(*args, **kwargs)
  return color_print

# Shortcut for colored print
for color in ["trace", "info", "success", "warning", "error"]:
  globals()[color] = _make_color_print(color)
def fatal(*args, with_traceback=False, **kwargs):
  """ Error colored print that calls 'exit(1)' instead of returning.
  Args:
    with_traceback Include a traceback after the message
    ...            Forwarded arguments
  """
  global error
  error(*args, **kwargs)
  if with_traceback:
    with Context("traceback", "trace"):
      traceback.print_exc()
  exit(1)

# Wrap the standard text output wrappers
sys.stdout = ContextIOWrapper(sys.stdout)
sys.stderr = ContextIOWrapper(sys.stderr)

# ---------------------------------------------------------------------------- #
# Uncaught exception context wrapping

def uncaught_wrap(hook):
  """ Wrap an uncaught hook with a context.
  Args:
    hook Uncaught hook to wrap
  Returns:
    Wrapped uncaught hook
  """
  def uncaught_call(etype, evalue, traceback):
    """ Update context, check if user exception or forward-call.
    Args:
      etype     Exception class
      evalue    Exception value
      traceback Traceback at the exception
    Returns:
      Forwarded value
    """
    if issubclass(etype, UserException):
      with Context("fatal", "error"):
        print(evalue)
    else:
      with Context("uncaught", "error"):
        return hook(etype, evalue, traceback)
  return uncaught_call

# Wrap the original exception hook
sys.excepthook = uncaught_wrap(sys.excepthook)

# ---------------------------------------------------------------------------- #
# Local module loading and post-processing

_imported = dict() # Map symbol name -> module source name

def import_exported_symbols(name, module, scope):
  """ Import the exported objects of the loaded module into the given scope.
  Args:
    name   Module name
    module Module instance
    scope  Target scope
  """
  global _imported
  if hasattr(module, "__all__"):
    for symname in getattr(module, "__all__"):
      # Check name
      if not hasattr(module, symname):
        with Context(None, "warning"):
          print("Symbol " + repr(symname) + " exported but not defined")
        continue
      if symname in _imported:
        with Context(None, "warning"):
          print("Symbol " + repr(symname) + " already exported by " + repr(_imported[symname]))
        continue
      if symname in scope:
        with Context(None, "warning"):
          print("Symbol " + repr(symname) + " already exported by '__init__.py'")
        continue
      # Import in module scope
      scope[symname] = getattr(module, symname)
      _imported[symname] = name

def import_directory(dirpath, scope, post=import_exported_symbols, ignore=["__init__"]):
  """ Import every module from the given directory in the given scope.
  Args:
    dirpath Directory path
    scope   Target scope
    post    Post module import function (name, module, scope) -> None
    ignore  List of module names to ignore
  """
  # Import in the scope of the caller
  for path in dirpath.iterdir():
    if path.is_file() and path.suffix == ".py":
      name = path.stem
      if "." in name or name in ignore:
        continue
      with Context(name, None):
        try:
          # Load module
          base = __import__(scope["__package__"], scope, scope, [name], 0)
          # Post processing
          if callable(post):
            post(name, getattr(base, name), scope)
        except Exception as err:
          with Context(None, "warning"):
            print("Loading failed for module " + repr(path.name) + ": " + str(err))
            with Context("traceback", "trace"):
              traceback.print_exc()

with Context("tools", None):
  import_directory(pathlib.Path(__file__).parent, globals())

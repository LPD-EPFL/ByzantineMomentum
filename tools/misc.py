# coding: utf-8
###
 # @file   misc.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2020 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Miscellaneous Python helpers.
###

__all__ = [
  "UnavailableException", "fatal_unavailable", "MethodCallReplicator", "ClassRegister",
  "parse_keyval", "fullqual", "onetime", "TimedContext", "interactive",
  "get_loaded_dependencies", "line_maximize", "pairwise"]

import os
import pathlib
import site
import sys
import threading
import time
import traceback

import tools

# ---------------------------------------------------------------------------- #
# Unavailable user exception class

def make_unavailable_exception_text(data, name, what="entry"):
  """ Make the explanatory string for an 'UnavailableException'.
  Args:
    data Iterable (over str) data set
    name Requested name in the data set
    what Textual description of what are the objects in the data set
  """
  # Preparation
  if len(data) == 0:
    end = "no %s available" % what
  else:
    sep = "%s· " % os.linesep
    end = "expected one of:%s%s" % (sep, sep.join(data))
  # Final string cat
  return "Unknown %s %r, %s" % (what, name, end)

def fatal_unavailable(*args, **kwargs):
  """ Helper forwarding the 'UnavailableException' explanatory string to 'fatal'.
  Args:
    ... Forward (keyword-)arguments to 'make_unavailable_exception_text'
  """
  tools.fatal(make_unavailable_exception_text(*args, **kwargs))

class UnavailableException(tools.UserException):
  """ Exception due to missing entry in a dictionary, where the entry is controlled by the user.
  """

  def __init__(self, *args, **kwargs):
    """ Error string constructor.
    Args:
      ... Forward (keyword-)arguments to 'make_unavailable_exception_text'
    """
    # Finalization
    self._text = make_unavailable_exception_text(*args, **kwargs)

  def __str__(self):
    """ Error to string conversion.
    Returns:
      Explanatory string
    """
    return self._text

# ---------------------------------------------------------------------------- #
# Simple method call replicator

class MethodCallReplicator:
  """ Simple method call replicator class.
  """

  def __init__(self, *args):
    """ Bind constructor.
    Args:
      ... Instance on which to replicate method calls (in the given order)
    """
    # Assertions
    assert len(args) > 0, "Expected at least one instance on which to forward method calls"
    # Finalization
    self.__instances = args

  def __getattr__(self, name):
    """ Returns a closure that replicate the method call.
    Args:
      name Name of the method
    Returns:
      Closure replicating the calls
    """
    # Target closures
    closures = [getattr(instance, name) for instance in self.__instances]
    # Replication closure
    def calls(*args, **kwargs):
      """ Simply replicate the calls, forwarding arguments.
      Args:
        ... Forwarded arguments
      Returns:
        List of returned values
      """
      return [closure(*args, **kwargs) for closure in closures]
    # Build the replication closure
    return calls

# ---------------------------------------------------------------------------- #
# Simple class register

class ClassRegister:
  """ Simple class register.
  """

  def __init__(self, singular, optplural=None):
    """ Denomination constructor.
    Args:
      singular  Singular denomination of the registered class
      optplural "Optional plural", e.g. "class(es)" for "class" (optional)
    """
    # Value deduction
    if optplural is None:
      optplural = singular + "(s)"
    # Finalization
    self.__denoms = (singular, optplural)
    self.__register = {}

  def itemize(self):
    """ Build an iterable over the available class names.
    Returns:
      Iterable over the available class names
    """
    return self.__register.keys()

  def register(self, name, cls):
    """ Register a new class.
    Args:
      name Class name
      cls  Associated class
    """
    # Assertions
    assert name not in self.__register, "Name " + repr(name) + " already in use while registering " + repr(getattr(cls, "__name__", "<unknown " + self.__denoms[0] + " class name>"))
    # Registering
    self.__register[name] = cls

  def instantiate(self, name, *args, **kwargs):
    """ Instantiate a registered class.
    Args:
      name Class name
      ...  Forwarded parameters
    Returns:
      Registered class instance
    """
    # Assertions
    if name not in self.__register:
      cause = "Unknown name " + repr(name) + ", "
      if len(self.__register) == 0:
        cause += "no registered " + self.__denoms[0]
      else:
        cause += "available " + self.__denoms[1] + ": '" + ("', '").join(self.__register.keys()) + "'"
      raise tools.UserException(cause)
    # Instantiation
    return self.__register[name](*args, **kwargs)

# ---------------------------------------------------------------------------- #
# Simple list of "<key>:<value>" into dictionary parser

def parse_keyval_auto_convert(val):
  """ Guess the type of the string representation, and return the converted value.
  Args:
    val Value to convert after type guessing
  Returns:
    Converted value, or same instance as 'val' if 'str' was the guessed type
  """
  # Try guess 'bool'
  low = val.lower()
  if low == "false":
    return False
  elif low == "true":
    return True
  # Try guess number
  for cls in (int, float):
    try:
      return cls(val)
    except ValueError:
      continue
  # Else guess string
  return val

def parse_keyval(list_keyval, defaults={}):
  """ Parse list of "<key>:<value>" into a dictionary.
  Args:
    list_keyval List of "<key>:<value>"
    defaults    Default key -> value to use (also ensure type, type is guessed for other keys)
  Returns:
    Associated dictionary
  """
  parsed = {}
  # Parsing
  sep = ":"
  for entry in list_keyval:
    pos = entry.find(sep)
    if pos < 0:
      raise tools.UserException("Expected list of " + repr("<key>:<value>") + ", got " + repr(entry) + " as one entry")
    key = entry[:pos]
    if key in parsed:
      raise tools.UserException("Key " + repr(key) + " had already been specified with value " + repr(parsed[key]))
    val = entry[pos + len(sep):]
    # Guess/assert type constructibility
    if key in defaults:
      try:
        cls = type(defaults[key])
        if cls is bool: # Special case
          val = val.lower() not in ("", "0", "n", "false")
        else:
          val = cls(val)
      except Exception:
        raise tools.UserException("Required key " + repr(key) + " expected a value of type " + repr(getattr(type(defaults[key]), "__name__", "<unknown>")))
    else:
      val = parse_keyval_auto_convert(val)
    # Bind (converted) value to associated key
    parsed[key] = val
  # Add default values (done first to be able to force a given type with 'required')
  for key in defaults:
    if key not in parsed:
      parsed[key] = defaults[key]
  # Return final dictionary
  return parsed

# ---------------------------------------------------------------------------- #
# Basic "full-qualification" string builder for a given instance/class

def fullqual(obj):
  """ Rebuild a string "qualifying" the given object for debugging purpose.
  Args:
    obj Object to "qualify"
  Returns:
    "Qualification", e.g.: 'tools.misc.fullqual' or 'instance of pathlib.Path'
  """
  # Prelude
  if isinstance(obj, type):
    prelude = ""
  else:
    prelude = "instance of "
    obj = type(obj)
  # Rebuilding
  return "%s%s.%s" % (prelude, getattr(obj, "__module__", "<unknown module>"), getattr(obj, "__qualname__", "<unknown name>"))

# ---------------------------------------------------------------------------- #
# Basic "full-qualification" string builder for a given instance/class

def onetime(name=None):
  """ Generate a one time-set (hidden) state variable getter and setter.
  Args:
    name Optional name of the global, onetime variable to access
  Returns:
    · (Threadsafe) getter closure
    · (Threadsafe) setter closure
  """
  global onetime_register
  # Check if name exists
  if name is not None and name in onetime_register:
    return onetime_register[name]
  # Private variables
  lock  = threading.Lock()
  value = False
  # Management closures
  def getter(*args, **kwargs):
    """ Check whether 'value' is set.
    Args:
      ... Ignored arguments
    Returns:
      Whether 'value' is set
    """
    nonlocal lock
    nonlocal value
    with lock:
      return value
  def setter(*args, **kwargs):
    """ Set 'value'.
    Args:
      ... Ignored arguments
    """
    nonlocal lock
    nonlocal value
    with lock:
      value = True
  # Register if need be, then return the management closures
  res = (getter, setter)
  if name is not None:
    onetime_register[name] = res
  return res

# Register for the onetime variables
onetime_register = dict()

# ---------------------------------------------------------------------------- #
# Plain context augmented with simple execution time measurement

class TimedContext(tools.Context):
  """ Timed context class.
  """

  def __init__(self, *args, **kwargs):
    """ Forward call to parent constructor.
    Args:
      ... Forwarded (keyword-)arguments
    """
    super().__init__(*args, **kwargs)

  def __enter__(self):
    """ Enter context: start chrono.
    Returns:
      Forwarded return value from parent
    """
    self._chrono = time.time()
    return super().__enter__()

  def __exit__(self, *args, **kwargs):
    """ Exit context: stop chrono and print elapsed time.
    Args:
      ... Forwarded arguments
    """
    # Measure elapsed time (in ns)
    elapsed = (time.time() - self._chrono) * 1000000000.
    # Print elapsed time
    for text in ["ns", "µs", "ms"]:
      if elapsed < 1000.:
        break
      elapsed /= 1000.
    else:
      text = "s"
    tools.trace("Execution time: %.3f %s" % (elapsed, text))
    # Forward call
    super().__exit__(*args, **kwargs)

# ---------------------------------------------------------------------------- #
# Switch to interactive mode, executing user inputs

def interactive(glbs=None, lcls=None, prompt=">>> ", cprmpt="... "):
  """ Switch to a simple interactive prompt, execute CTRL+D (or equivalent) to leave.
  Args:
    glbs   Globals dictionary to use, None to use caller's globals
    lcls   Locals dictionary to use, None to use given globals or caller's locals/globals
    prompt Command prompt to display
    cprmpt Command prompt to display when continuing a line
  """
  # Recover caller's globals and locals
  try:
    caller = sys._getframe().f_back
  except Exception:
    caller = None
    if glbs is None:
      tools.warning("Unable to recover caller's frame, locals and globals", context="interactive")
  if glbs is None:
    if caller is not None and hasattr(caller, "f_globals"):
      glbs = caller.f_globals
    else:
      glbs = dict()
  if lcls is None:
    if caller is not None and hasattr(caller, "f_locals"):
      lcls = caller.f_locals
    else:
      lcls = glbs
  # Command input and execution
  command   = ""
  statement = False
  while True:
    print(prompt if len(command) == 0 else cprmpt, end="", flush=True)
    try:
      # Input new line
      try:
        line = input()
        print("\033[A") # Trick to "advertise" new line on stdout after new line on stdin
      except BaseException as err:
        if any(isinstance(err, cls) for cls in (EOFError, KeyboardInterrupt)):
          print() # Since no new line was printed by pressing ENTER
        return
      # Handle expression
      if not statement:
        try:
          res = eval(line, glbs, lcls)
          if res is not None:
            print(res)
        except SyntaxError: # Heuristic that we are dealing with a statement
          statement = True
      # Handle single or multi-line statement(s)
      if statement:
        if len(command) == 0: # Just went through trying an expression
          command = line
          try:
            exec(command, glbs, lcls)
          except SyntaxError: # Heuristic that we are dealing with a multi-line statement
            continue
        elif len(line) > 0:
          command += os.linesep + line
          continue
        else: # Multi-line statement is complete
          exec(command, glbs, lcls)
    except Exception:
      with tools.Context("uncaught", "error"):
        traceback.print_exc()
    command = ""
    statement = False

# ---------------------------------------------------------------------------- #
# List non-standard, currently loaded module names and metadata.

def get_loaded_dependencies():
  """ List non-builtin, currently loaded root module names and metadata.
  Returns:
    List of tuples (<root module name>, <version or 'None'>, <0: is standard, 1: is site-specific, 2: is local>)
  Raises:
    'RuntimeError' on unsupported platforms
  """
  # Get the site-packages directories, and make "flavor"-detection closure
  path_sites = tuple(pathlib.Path(path) for path in site.getsitepackages() + [site.getusersitepackages()])
  def flavor_of(path):
    path = pathlib.Path(path)
    for path_site in path_sites:
      try:
        path.relative_to(path_site)
        return get_loaded_dependencies.IS_SITE
      except ValueError:
        pass
    for path_site in path_sites:
      try:
        path.relative_to(path_site.parent)
        return get_loaded_dependencies.IS_STANDARD
      except ValueError:
        pass
    return get_loaded_dependencies.IS_LOCAL
  # Iterate over the loaded modules
  res = list()
  for name, module in sys.modules.items():
    # Skip non-root modules
    if "." in name:
      continue
    # Get module path (and so skip built-in modules)
    path = getattr(module, "__file__", None)
    if path is None:
      continue
    # Get module version (if any)
    version = getattr(module, "__version__", None)
    # Get module "flavor"
    flavor = flavor_of(path)
    # Store entry
    res.append((name, version, flavor))
  # Return found root modules
  return res

# Register constants
get_loaded_dependencies.IS_STANDARD = 0
get_loaded_dependencies.IS_SITE     = 1
get_loaded_dependencies.IS_LOCAL    = 2

# ---------------------------------------------------------------------------- #
# Find the x maximizing a function y = f(x), with (x, y) ∊ ℝ⁺× ℝ

def line_maximize(scape, evals=16, start=0., delta=1., ratio=0.8):
  """ Best-effort arg-maximize a scape: ℝ⁺⟶ ℝ, by mere exploration.
  Args:
    scape Function to best-effort arg-maximize
    evals Maximum number of evaluations, must be a positive integer
    start Initial x evaluated, must be a non-negative float
    delta Initial step delta, must be a positive float
    ratio Contraction ratio, must be between 0.5 and 1. (both excluded)
  Returns:
    Best-effort maximizer x under the evaluation budget
  """
  # Variable setup
  best_x = start
  best_y = scape(best_x)
  evals -= 1
  # Expansion phase
  while evals > 0:
    prop_x = best_x + delta
    prop_y = scape(prop_x)
    evals -= 1
    # Check if best
    if prop_y > best_y:
      best_y = prop_y
      best_x = prop_x
      delta *= 2
    else:
      delta *= ratio
      break
  # Contraction phase
  while evals > 0:
    if prop_x < best_x:
      prop_x += delta
    else:
      x = prop_x - delta
      while x < 0:
        x = (x + prop_x) / 2
      prop_x = x
    prop_y = scape(prop_x)
    evals -= 1
    # Check if best
    if prop_y > best_y:
      best_y = prop_y
      best_x = prop_x
    # Reduce delta
    delta *= ratio
  # Return found maximizer
  return best_x

# ---------------------------------------------------------------------------- #
# Simple generator on the pairs (x, y) of an indexable such that index x < index y

def pairwise(data):
  """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
  Args:
    data Indexable (including ability to query length) containing the elements
  Returns:
    Generator over the pairs of the elements of 'data'
  """
  n = len(data)
  for i in range(n - 1):
    for j in range(i + 1, n):
      yield (data[i], data[j])

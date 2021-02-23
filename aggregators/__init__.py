# coding: utf-8
###
 # @file   __init__.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Loading of the local modules.
 #
 # Each rule MUST support taking any named arguments, possibly ignoring them.
 # The parameters MUST all be passed as their keyword arguments.
 # The reserved argument names, and their interface, are the following:
 # · gradients: Non-empty list of gradients to aggregate
 # · f        : Number of Byzantine gradients to support
 # · model    : Model (duck-typing 'experiments.Model') with valid default dataset and loss set
 # The rule, given "valid" parameter(s), MUST NOT return a tensor that is a reference to any tensor given as parameter.
 #
 # Each rule MUST provide a "check" member function, taking the same arguments as the rule itself.
 # The "check" member function returns 'None' when the parameters are valid,
 # or an explanatory string when the parameters are not valid.
 # The check member function MUST NOT modify the given parameters.
 #
 # Once registered, the check member function will be available as member "check".
 # The raw function and a wrapped checking the input/output of the raw function
 # will respectively be available as members "unchecked" and "checked".
 # Which of these two functions is called by default depends whether debug mode is enabled.
###

import pathlib
import torch

import tools

# ---------------------------------------------------------------------------- #
# Automated GAR loader

def make_gar(unchecked, check, upper_bound=None, influence=None):
  """ GAR wrapper helper.
  Args:
    unchecked   Associated function (see module description)
    check       Parameter validity check function
    upper_bound Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm to use this aggregation rule: (n, f, d) -> float
    influence   Attack acceptation ratio function
  Returns:
    Wrapped GAR
  """
  # Closure wrapping the call with checks
  def checked(**kwargs):
    # Check parameter validity
    message = check(**kwargs)
    if message is not None:
      raise tools.UserException(f"Aggregation rule {name!r} cannot be used with the given parameters: {message}")
    # Aggregation (hard to assert return value, duck-typing is allowed...)
    return unchecked(**kwargs)
  # Select which function to call by default
  func = checked if __debug__ else unchecked
  # Bind all the (sub) functions to the selected function
  setattr(func, "check", check)
  setattr(func, "checked", checked)
  setattr(func, "unchecked", unchecked)
  setattr(func, "upper_bound", upper_bound)
  setattr(func, "influence", influence)
  # Return the selected function with the associated name
  return func

def register(name, unchecked, check, upper_bound=None, influence=None):
  """ Simple registration-wrapper helper.
  Args:
    name        GAR name
    unchecked   Associated function (see module description)
    check       Parameter validity check function
    upper_bound Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm to use this aggregation rule: (n, f, d) -> float
    influence   Attack acceptation ratio function
  """
  global gars
  # Check if name already in use
  if name in gars:
    tools.warning(f"Unable to register {name!r} GAR: name already in use")
    return
  # Export the selected function with the associated name
  gars[name] = make_gar(unchecked, check, upper_bound=upper_bound, influence=influence)

# Registered rules (mapping name -> aggregation rule)
gars = dict()

# Load all local modules
with tools.Context("aggregators", None):
  tools.import_directory(pathlib.Path(__file__).parent, globals())

# Bind/overwrite the GAR name with the associated rules in globals()
for name, rule in gars.items():
  globals()[name] = rule

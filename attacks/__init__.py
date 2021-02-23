# coding: utf-8
###
 # @file   __init__.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Loading of the local modules.
 #
 # Each attack MUST support taking any named arguments, possibly ignoring them.
 # The parameters MUST all be passed as their keyword arguments.
 # The reserved argument names, and their interface, are the following:
 # · grad_honest: Non-empty list of honest gradients generated
 # · f_decl     : Number of declared Byzantine gradients at the GAR
 # · f_real     : Number of actual Byzantine gradients to generate
 # · model      : Model (duck-typing 'experiments.Model') with valid default dataset and loss set
 # · defense    : Aggregation rule (see module 'aggregators') in use to defeat
 # The attack, given "valid" parameter(s), MUST return a list of f_byz tensor(s).
 # Each of these returned tensors MUST NOT be a reference to any tensor given as parameter,
 # although each returned tensors MAY be references to the same tensor.
 #
 # Each attack MUST provide a "check" function, taking the same arguments as the attack itself.
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
# Automated attack loader

def register(name, unchecked, check):
  """ Simple registration-wrapper helper.
  Args:
    name      Attack name
    unchecked Associated function (see module description)
    check     Parameter validity check function
  """
  global attacks
  # Check if name already in use
  if name in attacks:
    tools.warning(f"Unable to register {name!r} attack: name already in use")
    return
  # Closure wrapping the call with checks
  def checked(f_real, **kwargs):
    # Check parameter validity
    message = check(f_real=f_real, **kwargs)
    if message is not None:
      raise tools.UserException(f"Attack {name!r} cannot be used with the given parameters: {message}")
    # Attack
    res = unchecked(f_real=f_real, **kwargs)
    # Forward asserted return value
    assert isinstance(res, list) and len(res) == f_real, f"Expected attack {name!r} to return a list of {f_real} Byzantine gradients, got {res!r}"
    return res
  # Select which function to call by default
  func = checked if __debug__ else unchecked
  # Bind all the (sub) functions to the selected function
  setattr(func, "check", check)
  setattr(func, "checked", checked)
  setattr(func, "unchecked", unchecked)
  # Export the selected function with the associated name
  attacks[name] = func

# Registered attacks (mapping name -> attack)
attacks = dict()

# Load native and all local modules
with tools.Context("attacks", None):
  tools.import_directory(pathlib.Path(__file__).parent, globals())

# Bind/overwrite the attack names with the associated attacks in globals()
for name, attack in attacks.items():
  globals()[name] = attack

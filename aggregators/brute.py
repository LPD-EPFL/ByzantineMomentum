# coding: utf-8
###
 # @file   brute.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Brute GAR.
###

import tools
from . import register

import itertools
import math
import torch

# Optional 'native' module
try:
  import native
except ImportError:
  native = None

# ---------------------------------------------------------------------------- #
# Brute GAR

def _compute_selection(gradients, f, **kwargs):
  """ Brute rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    Selection index set
  """
  n = len(gradients)
  # Compute all pairwise distances
  distances = [0] * (n * (n - 1) // 2)
  for i, (x, y) in enumerate(tools.pairwise(tuple(range(n)))):
    distances[i] = gradients[x].sub(gradients[y]).norm().item()
  # Select the set of smallest diameter
  sel_iset = None
  sel_diam = None
  for cur_iset in itertools.combinations(range(n), n - f):
    # Compute the current diameter (max of pairwise distances)
    cur_diam = 0.
    for x, y in tools.pairwise(cur_iset):
      # Get distance between these two gradients ("magic" formula valid since x < y)
      cur_dist = distances[(2 * n - x - 3) * x // 2 + y - 1]
      # Check finite distance (non-Byzantine gradient must only contain finite coordinates), drop set if non-finite
      if not math.isfinite(cur_dist):
        break
      # Check if new maximum
      if cur_dist > cur_diam:
        cur_diam = cur_dist
    else:
      # Check if new selected diameter
      if sel_iset is None or cur_diam < sel_diam:
        sel_iset = cur_iset
        sel_diam = cur_diam
  # Return the selected gradients
  assert sel_iset is not None, "Too many non-finite gradients: a non-Byzantine gradient must only contain finite coordinates"
  return sel_iset

def aggregate(gradients, f, **kwargs):
  """ Brute rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  sel_iset = _compute_selection(gradients, f, **kwargs)
  return sum(gradients[i] for i in sel_iset).div_(len(gradients) - f)

def aggregate_native(gradients, f, **kwargs):
  """ Brute rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  return native.brute.aggregate(gradients, f)

def check(gradients, f, **kwargs):
  """ Check parameter validity for Brute rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
  if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 1:
    return f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 1 ≤ f ≤ {(len(gradients) - 1) // 2}"

def upper_bound(n, f, d):
  """ Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm to use this rule.
  Args:
    n Number of workers (Byzantine + non-Byzantine)
    f Expected number of Byzantine workers
    d Dimension of the gradient space
  Returns:
    Theoretical upper-bound
  """
  return (n - f) / (math.sqrt(8) * f)

def influence(honests, attacks, f, **kwargs):
  """ Compute the ratio of accepted Byzantine gradients.
  Args:
    honests Non-empty list of honest gradients to aggregate
    attacks List of attack gradients to aggregate
    f       Number of Byzantine gradients to tolerate
    m       Optional number of averaged gradients for Multi-Krum
    ...     Ignored keyword-arguments
  Returns:
    Ratio of accepted
  """
  gradients = honests + attacks
  # Compute the selection set
  sel_iset = _compute_selection(gradients, f, **kwargs)
  # Compute the influence ratio
  count = 0
  for i in sel_iset:
    gradient = gradients[i]
    for attack in attacks:
      if gradient is attack:
        count += 1
        break
  return count / (len(gradients) - f)

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (pytorch version)
method_name = "brute"
register(method_name, aggregate, check, upper_bound=upper_bound, influence=influence)

# Register aggregation rule (native version, if available)
if native is not None:
  native_name = method_name
  method_name = "native-" + method_name
  if native_name in dir(native):
    register(method_name, aggregate_native, check, upper_bound)
  else:
    tools.warning(f"GAR {method_name!r} could not be registered since the associated native module {native_name!r} is unavailable")

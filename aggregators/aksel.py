# coding: utf-8
###
 # @file   aksel.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2020-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Aksel GAR.
###

import tools
from . import register

import torch

# ---------------------------------------------------------------------------- #
# Aksel GAR

def _compute_distances(gradients, f, mode, **kwargs):
  """ Aksel rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    mode      Operation mode, one of: 'mid', 'n-f'
    ...       Ignored keyword-arguments
  Returns:
    List of (gradient, distance to the median) sorted by increasing distance,
    Number of gradients to aggregate
  """
  n = len(gradients)
  # Prepare gradients
  g = torch.stack(gradients)
  # Compute median
  m = g.median(dim=0)[0]
  # Measure squared distances to median
  d = list((i, (x - m).pow_(2).sum().item()) for i, x in enumerate(gradients))
  # Average closest to median according to mode
  if mode == "mid":
    c = (n + 1) // 2
  elif mode == "n-f":
    c = n - f
  else:
    raise NotImplementedError
  d.sort(key=lambda x: x[1])
  return d, c

def aggregate(gradients, f, mode="mid", **kwargs):
  """ Aksel rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    mode      Operation mode, see: '_compute_distances'
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  # Compute distances and aggregate
  d, c = _compute_distances(gradients, f, mode, **kwargs)
  return sum(gradients[i] for i, _ in d[:c]).div_(c)

def check(gradients, f, mode="mid", **kwargs):
  """ Check parameter validity for Aksel rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    mode      Operation mode, one of: 'mid', 'n-f'
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
  if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f + 1:
    return f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 1 ≤ f ≤ {(len(gradients) - 1) // 2}"
  if mode not in ("mid", "n-f"):
    return f"Invalid operation mode {mode!r}"

def influence(honests, attacks, f, mode="mid", **kwargs):
  """ Compute the ratio of accepted Byzantine gradients.
  Args:
    honests Non-empty list of honest gradients to aggregate
    attacks List of attack gradients to aggregate
    f       Number of Byzantine gradients to tolerate
    mode    Operation mode, see 'aggregate'
    ...     Ignored keyword-arguments
  Returns:
    Ratio of accepted attack gradients
  """
  gradients = honests + attacks
  # Compute sorted distances and aggregation count
  d, c = _compute_distances(gradients, f, mode, **kwargs)
  # Compute the influence ratio
  count = 0
  for i, _ in d[:c]:
    gradient = gradients[i]
    for attack in attacks:
      if gradient is attack:
        count += 1
        break
  return count / c

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule
register("aksel", aggregate, check, influence=influence)

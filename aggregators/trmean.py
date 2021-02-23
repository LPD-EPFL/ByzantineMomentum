# coding: utf-8
###
 # @file   trmean.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2020-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Trimmed mean, Phocas and MeaMed GARs.
###

import tools
from . import register

import torch

# ---------------------------------------------------------------------------- #
# Common operations

def trmean(g, f):
  """ Trimmed mean rule.
  Args:
    g Non-empty stack of gradients to aggregate
    f Number of Byzantine gradients to tolerate
  Returns:
    Aggregated gradient
  """
  # Compute average of "inner" values
  return g.sort(dim=0).values[f:-f].mean(dim=0)

def closest(g, f, c):
  """ Select the coordinate-wise average of the 'g.shape[0] - f' closest values to 'c'.
  Args:
    g Non-empty stack of gradients to aggregate
    f Number of Byzantine gradients to tolerate
    c "Center" gradient to use as reference
  Returns:
    Coordinate-wise gradient as described above
  """
  # Recover constants
  n, d = g.shape
  m = n - f
  # Compute aggregated gradient
  p = g.clone().sub_(c).abs_().topk(m, dim=0, largest=False, sorted=False).indices
  p.mul_(d).add_(torch.arange(0, d, dtype=p.dtype, device=p.device))
  return g.take(p).mean(dim=0)

def check(gradients, f, **kwargs):
  """ Check parameter validity for the trimmed mean rule.
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

# ---------------------------------------------------------------------------- #
# GARs

def aggr_trmean(gradients, f, **kwargs):
  """ Trimmed mean rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  # Compute trimmed mean
  return trmean(torch.stack(gradients), f)

def aggr_phocas(gradients, f, **kwargs):
  """ Phocas rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  # Stack gradients
  g = torch.stack(gradients)
  # Compute Phocas
  c = trmean(g, f)
  return closest(g, f, c)

def aggr_meamed(gradients, f, **kwargs):
  """ MeaMed rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  # Stack gradients
  g = torch.stack(gradients)
  # Compute MeaMed
  c = g.median(dim=0).values
  return closest(g, f, c)

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule
register("trmean", aggr_trmean, check)
register("phocas", aggr_phocas, check)
register("meamed", aggr_meamed, check)

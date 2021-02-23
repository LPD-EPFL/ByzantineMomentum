# coding: utf-8
###
 # @file   cge.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Comparative Gradient Elimination (CGE) GAR.
 #
 # This algorithm has been introduced in the following paper:
 #   Approximate Byzantine Fault-Tolerance in Distributed Optimization.
 #   Shuo Liu, Nirupam Gupta, Nitin H. Vaidya.
 #   Arxiv, 22 Jan 2021.
###

import math

from . import register

# ---------------------------------------------------------------------------- #
# Comparative Gradient Elimination (CGE) GAR

def _compute_normed(grads):
  """ Compute norms and sort gradients by increasing norm, handling non-finite coordinates as belonging to Byzantine gradients.
  Args:
    grads Iterable of gradients
  Returns:
    List of gradients sorted by increasing norm
  """
  def byznorm(grad):
    norm = grad.norm().item()
    return norm if math.isfinite(norm) else math.inf
  return sorted(((byznorm(grad), grad) for grad in grads), key=lambda pair: pair[0])

def aggregate(gradients, f, **kwargs):
  """ Comparative Gradient Elimination (CGE) rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  m = len(gradients) - f
  # Compute norms and sort gradients by increasing norm
  normed = _compute_normed(gradients)
  # Compute and return the average of the m = n - f smallest-norm gradients
  res = normed[0][1].clone().detach_()
  for _, gradient in normed[1:m]:
    res.add_(gradient)
  res.div_(m)
  return res

def check(gradients, f, m=None, **kwargs):
  """ Check parameter validity for Multi-Krum rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    m         Optional number of averaged gradients for Multi-Krum
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"

def influence(honests, attacks, f, **kwargs):
  """ Compute the ratio of accepted Byzantine gradients.
  Args:
    honests Non-empty list of honest gradients to aggregate
    attacks List of attack gradients to aggregate
    f       Number of Byzantine gradients to tolerate
    ...     Ignored keyword-arguments
  Returns:
    Ratio of accepted
  """
  gradients = honests + attacks
  m = len(gradients) - f
  # Compute norms and sort gradients by increasing norm
  normed = _compute_normed(gradients)
  # Compute the influence ratio
  count = 0
  for _, gradient in normed[:m]:
    for attack in attacks:
      if gradient is attack:
        count += 1
        break
  return count / m

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (PyTorch version)
register("cge", aggregate, check, influence=influence)

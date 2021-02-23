# coding: utf-8
###
 # @file   average.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Simple average GAR.
###

from . import register

# ---------------------------------------------------------------------------- #
# Average GAR

def aggregate(gradients, **kwargs):
  """ Averaging rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    Average gradient
  """
  return sum(gradients) / len(gradients)

def check(gradients, **kwargs):
  """ Check parameter validity for the averaging rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    None if valid, otherwise error message string
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"

def influence(honests, attacks, **kwargs):
  """ Compute the ratio of accepted Byzantine gradients.
  Args:
    honests Non-empty list of honest gradients to aggregate
    attacks List of attack gradients to aggregate
    ...     Ignored keyword-arguments
  """
  return len(attacks) / (len(honests) + len(attacks))

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule
register("average", aggregate, check, influence=influence)

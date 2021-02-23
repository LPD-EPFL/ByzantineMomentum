# coding: utf-8
###
 # @file   template.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2020-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Template code for a GAR.
 # See `__init__.py` docstring for the interface.
###

from . import register

# ---------------------------------------------------------------------------- #
# GAR template

def aggregate(gradients, f, **kwargs):
  """ Multi-Krum rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    Aggregated gradient
  """
  raise NotImplementedError("I am template code, please replace me with useful stuff")

def check(gradients, f, **kwargs):
  """ Check parameter validity for Multi-Krum rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    f         Number of Byzantine gradients to tolerate
    ...       Ignored keyword-arguments
  Returns:
    Whether the given parameters are valid for this rule
  """
  return "I am template code, you should not be using me"

def upper_bound(n, f, d):
  """ Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm to use this rule.
  Args:
    n Number of workers (Byzantine + non-Byzantine)
    f Expected number of Byzantine workers
    d Dimension of the gradient space
  Returns:
    Theoretical upper-bound
  """
  raise NotImplementedError("I am optional (but still template) code, please replace me with useful stuff or delete me")

# ---------------------------------------------------------------------------- #
# GAR registering

# Register the GAR
register("template", aggregate, check, upper_bound)

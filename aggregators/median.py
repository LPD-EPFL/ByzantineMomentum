# coding: utf-8
###
 # @file   median.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2020 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # NaN-resilient, coordinate-wise median GAR.
###

import tools
from . import register

import math
import torch

# Optional 'native' module
try:
  import native
except ImportError:
  native = None

# ---------------------------------------------------------------------------- #
# NaN-resilient, coordinate-wise median GAR

def aggregate(gradients, **kwargs):
  """ NaN-resilient median coordinate-per-coordinate rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    NaN-resilient, coordinate-wise median of the gradients
  """
  return torch.stack(gradients).median(dim=0)[0]

def aggregate_native(gradients, **kwargs):
  """ NaN-resilient median coordinate-per-coordinate rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    NaN-resilient, coordinate-wise median of the gradients
  """
  return native.median.aggregate(gradients)

def check(gradients, **kwargs):
  """ Check parameter validity for the median rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    Whether the given parameters are valid for this rule
  """
  if not isinstance(gradients, list) or len(gradients) < 1:
    return "Expected a list of at least one gradient to aggregate, got %r" % gradients

def upper_bound(n, f, d):
  """ Compute the theoretical upper bound on the ratio non-Byzantine standard deviation / norm to use this rule.
  Args:
    n Number of workers (Byzantine + non-Byzantine)
    f Expected number of Byzantine workers
    d Dimension of the gradient space
  Returns:
    Theoretical upper-bound
  """
  return 1 / math.sqrt(n - f)

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (pytorch version)
method_name = "median"
register(method_name, aggregate, check, upper_bound)

# Register aggregation rule (native version, if available)
if native is not None:
  native_name = method_name
  method_name = "native-" + method_name
  if native_name in dir(native):
    register(method_name, aggregate_native, check, upper_bound)
  else:
    tools.warning("GAR %r could not be registered since the associated native module %r is unavailable" % (method_name, native_name))

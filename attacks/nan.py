# coding: utf-8
###
 # @file   nan.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Attack that generates NaN gradient(s), hence the name.
###

import math
import torch

from . import register

# ---------------------------------------------------------------------------- #
# Non-finite gradient attack

def attack(grad_honests, f_real, **kwargs):
  """ Generate non-finite gradients.
  Args:
    grad_honests Non-empty list of honest gradients
    f_real       Number of Byzantine gradients to generate
    ...          Ignored keyword-arguments
  Returns:
    Generated Byzantine gradients
  """
  # Fast path
  if f_real == 0:
    return list()
  # Generate the non-finite Byzantine gradient
  byz_grad = torch.empty_like(grad_honests[0])
  byz_grad.copy_(torch.tensor((math.nan,), dtype=byz_grad.dtype))
  # Return this Byzantine gradient 'f_real' times
  return [byz_grad] * f_real

def check(grad_honests, f_real, **kwargs):
  """ Check parameter validity for this attack.
  Args:
    grad_honests Non-empty list of honest gradients
    f_real       Number of Byzantine gradients to generate
    ...          Ignored keyword-arguments
  Returns:
    Whether the given parameters are valid for this attack
  """
  if not isinstance(grad_honests, list) or len(grad_honests) == 0:
    return f"Expected a non-empty list of honest gradients, got {grad_honests!r}"
  if not isinstance(f_real, int) or f_real < 0:
    return f"Expected a non-negative number of Byzantine gradients to generate, got {f_real!r}"

# ---------------------------------------------------------------------------- #
# Attack registering

# Register the attack
register("nan", attack, check)

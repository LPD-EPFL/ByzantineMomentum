# coding: utf-8
###
 # @file   identical.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2020 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Collection of attacks which submit f identical gradients, which consist in
 # adding as much of one attack vector to the average of the honest gradients.
 #
 # These attacks have been introduced in/adapted from the following papers:
 # bulyan · El Mhamdi El Mahdi, Guerraoui Rachid, and Rouault Sébastien.
 #          The Hidden Vulnerability of Distributed Learning in Byzantium.
 #          ICML 2018. URL: http://proceedings.mlr.press/v80/mhamdi18a.html
 # empire · Cong Xie, Oluwasanmi Koyejo, Indranil Gupta.
 #          Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation.
 #          UAI 2019. URL: http://auai.org/uai2019/proceedings/papers/83.pdf
 # little · Moran Baruch, Gilad Baruch, Yoav Goldberg.
 #          A Little Is Enough: Circumventing Defenses For Distributed Learning.
 #          2019 Feb 16. ArXiv. URL: https://arxiv.org/pdf/1902.06156v1
###

import tools

import math
import torch

from . import register

# ---------------------------------------------------------------------------- #
# Generic attack implementation generator

def make_attack(compute_direction):
  """ Make the attack gradient generation closure associated with an attack direction.
  Args:
    compute_direction Attack vector computation, (stacked honest gradients, average honest gradient, forwarded keyword-arguments...) -> attack vector (in the gradient space, no reference)
  Returns:
    Byzantine gradient generation closure
  """
  def attack(grad_honests, f_real, f_decl, defense, model, factor=-16, negative=False, **kwargs):
    """ Generate the attack gradients.
    Args:
      grad_honests Non-empty list of honest gradients
      f_decl       Number of declared Byzantine gradients
      f_real       Number of Byzantine gradients to generate
      defense      Aggregation rule in use to defeat
      model        Model with valid default dataset and loss set
      factor       Fixed attack factor if positive, number of evaluations for best attack factor if negative
      negative     Use a negative factor instead of a positive one
      ...          Forwarded keyword-arguments
    Returns:
      Generated Byzantine gradients (all references to one)
    """
    # Fast path
    if f_real == 0:
      return list()
    # Stack and compute the average honest gradient, and then the attack vector
    grad_stck = torch.stack(grad_honests)
    grad_avg  = grad_stck.mean(dim=0)
    grad_att  = compute_direction(grad_stck, grad_avg, **kwargs)
    # Evaluate the best attack factor (if required)
    if factor < 0:
      def eval_factor(factor):
        # Apply the given factor
        if negative:
          factor = -factor
        grad_attack = grad_avg + factor * grad_att
        # Measure effective squared distance
        aggregated = defense(gradients=(grad_honests + [grad_attack] * f_real), f=f_decl, model=model)
        aggregated.sub_(grad_avg)
        return aggregated.dot(aggregated).item()
      factor = tools.line_maximize(eval_factor, evals=math.ceil(-factor))
    else:
      if negative:
        factor = -factor
    # Generate the Byzantine gradient from the given/computed factor
    byz_grad = grad_avg
    grad_att.mul_(factor)
    byz_grad.add_(grad_att)
    # Return this Byzantine gradient 'f_real' times
    return [byz_grad] * f_real
  # Return the attack closure
  return attack

def check(grad_honests, f_real, defense, factor=-16, negative=False, **kwargs):
  """ Check parameter validity for this attack template.
  Args:
    grad_honests Non-empty list of honest gradients
    f_real       Number of Byzantine gradients to generate
    defense      Aggregation rule in use to defeat
    ...          Ignored keyword-arguments
  Returns:
    Whether the given parameters are valid for this attack
  """
  if not isinstance(grad_honests, list) or len(grad_honests) == 0:
    return "Expected a non-empty list of honest gradients, got %r" % (grad_honests,)
  if not isinstance(f_real, int) or f_real < 0:
    return "Expected a non-negative number of Byzantine gradients to generate, got %r" % (f_real,)
  if not callable(defense):
    return "Expected a callable for the aggregation rule, got %r" % (defense,)
  if not ((isinstance(factor, float) and factor > 0) or (isinstance(factor, int) and factor != 0)):
    return "Expected a positive number or a negative integer for the attack factor, got %r" % (factor,)
  if not isinstance(negative, bool):
    return "Expected a boolean for optional parameter 'negative', got %r" % (negative,)

# ---------------------------------------------------------------------------- #
# Attack vector computations

def bulyan(grad_stck, grad_avg, target_idx=-1, **kwargs):
  """ Compute the attack vector adapted from "The Hidden Vulnerability".
  Args:
    target_idx Index of the targeted coordinate, "all" for all
  See:
    make_attack
  """
  if target_idx == "all":
    return torch.ones_like(grad_avg)
  else:
    assert isinstance(target_idx, int), "Expected an integer or \"all\" for 'target_idx', got %r" % (target_idx,)
    grad_att = torch.zeros_like(grad_avg)
    grad_att[target_idx] = 1
    return grad_att

def empire(grad_stck, grad_avg, **kwargs):
  """ Compute the attack vector adapted from "Fall of Empires".
  See:
    make_attack
  """
  return grad_avg.neg()

def little(grad_stck, grad_avg, **kwargs):
  """ Compute the attack vector adapted from "A Little is Enough".
  See:
    make_attack
  """
  return grad_stck.var(dim=0).sqrt_()

# ---------------------------------------------------------------------------- #
# Attack registrations

# Register the attacks
for name, func in (("bulyan", bulyan), ("empire", empire), ("little", little)):
  register(name, make_attack(func), check)

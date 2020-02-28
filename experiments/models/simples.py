# coding: utf-8
###
 # @file   simples.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2020 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Collection of simple models.
###

__all__ = ["full", "conv"]

import torch

# ---------------------------------------------------------------------------- #
# Simple fully-connected model, for MNIST

class _Full(torch.nn.Module):
  """ Simple, small fully connected model.
  """

  def __init__(self):
    """ Model parameter constructor.
    """
    super().__init__()
    # Build parameters
    self._f1 = torch.nn.Linear(28 * 28, 100)
    self._f2 = torch.nn.Linear(100, 10)

  def forward(self, x):
    """ Model's forward pass.
    Args:
      x Input tensor
    Returns:
      Output tensor
    """
    # Forward pass
    x = torch.nn.functional.relu(self._f1(x.view(-1, 28 * 28)))
    x = torch.nn.functional.log_softmax(self._f2(x), dim=1)
    return x

def full(*args, **kwargs):
  """ Build a new simple, fully connected model.
  Args:
    ... Forwarded (keyword-)arguments
  Returns:
    Fully connected model
  """
  global _Full
  return _Full(*args, **kwargs)

# ---------------------------------------------------------------------------- #
# Simple convolutional model, for MNIST

class _Conv(torch.nn.Module):
  """ Simple, small convolutional model.
  """

  def __init__(self):
    """ Model parameter constructor.
    """
    super().__init__()
    # Build parameters
    self._c1 = torch.nn.Conv2d(1, 20, 5, 1)
    self._c2 = torch.nn.Conv2d(20, 50, 5, 1)
    self._f1 = torch.nn.Linear(800, 500)
    self._f2 = torch.nn.Linear(500, 10)

  def forward(self, x):
    """ Model's forward pass.
    Args:
      x Input tensor
    Returns:
      Output tensor
    """
    # Forward pass
    x = torch.nn.functional.relu(self._c1(x))
    x = torch.nn.functional.max_pool2d(x, 2, 2)
    x = torch.nn.functional.relu(self._c2(x))
    x = torch.nn.functional.max_pool2d(x, 2, 2)
    x = torch.nn.functional.relu(self._f1(x.view(-1, 800)))
    x = torch.nn.functional.log_softmax(self._f2(x), dim=1)
    return x

def conv(*args, **kwargs):
  """ Build a new simple, convolutional model.
  Args:
    ... Forwarded (keyword-)arguments
  Returns:
    Convolutional model
  """
  global _Conv
  return _Conv(*args, **kwargs)

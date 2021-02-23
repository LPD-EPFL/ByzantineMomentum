# coding: utf-8
###
 # @file   simples.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Collection of simple models.
###

__all__ = ["full", "conv", "logit", "linear"]

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

# ---------------------------------------------------------------------------- #
# Simple(r) logistic regression model

class _Logit(torch.nn.Module):
  """ Simple logistic regression model.
  """

  def __init__(self, din, dout=1):
    """ Model parameter constructor.
    Args:
      din  Number of input dimensions
      dout Number of output dimensions
    """
    super().__init__()
    # Store model parameters
    self._din  = din
    self._dout = dout
    # Build parameters
    self._linear = torch.nn.Linear(din, dout)

  def forward(self, x):
    """ Model's forward pass.
    Args:
      x Input tensor
    Returns:
      Output tensor
    """
    return torch.sigmoid(self._linear(x.view(-1, self._din)))

def logit(*args, **kwargs):
  """ Build a new simple, fully connected model.
  Args:
    ... Forwarded (keyword-)arguments
  Returns:
    Fully connected model
  """
  global _Logit
  return _Logit(*args, **kwargs)

# ---------------------------------------------------------------------------- #
# Simple(st) linear model

class _Linear(torch.nn.Module):
  """ Simple linear model.
  """

  def __init__(self, din, dout=1):
    """ Model parameter constructor.
    Args:
      din  Number of input dimensions
      dout Number of output dimensions
    """
    super().__init__()
    # Store model parameters
    self._din  = din
    self._dout = dout
    # Build parameters
    self._linear = torch.nn.Linear(din, dout)

  def forward(self, x):
    """ Model's forward pass.
    Args:
      x Input tensor
    Returns:
      Output tensor
    """
    return self._linear(x.view(-1, self._din))

def linear(*args, **kwargs):
  """ Build a new simple, fully connected model.
  Args:
    ... Forwarded (keyword-)arguments
  Returns:
    Fully connected model
  """
  global _Linear
  return _Linear(*args, **kwargs)

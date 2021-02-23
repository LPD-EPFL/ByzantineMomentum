# coding: utf-8
###
 # @file   empire.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # The model from "Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation".
 # (The original paper did not include the CIFAR-100 variant.)
###

__all__ = ["cnn"]

import torch

# ---------------------------------------------------------------------------- #
# Simple convolutional model, for CIFAR-10/100 (3 input channels)

class _CNN(torch.nn.Module):
  """ Simple, small convolutional model.
  """

  def __init__(self, cifar100=False):
    """ Model parameter constructor.
    Args:
        cifar100 Build the CIFAR-100 variant (instead of the CIFAR-10)
    """
    super().__init__()
    # Build parameters
    self._c1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
    self._b1 = torch.nn.BatchNorm2d(self._c1.out_channels)
    self._c2 = torch.nn.Conv2d(self._c1.out_channels, 64, kernel_size=3, padding=1)
    self._b2 = torch.nn.BatchNorm2d(self._c2.out_channels)
    self._m1 = torch.nn.MaxPool2d(2)
    self._d1 = torch.nn.Dropout(p=0.25)
    self._c3 = torch.nn.Conv2d(self._c2.out_channels, 128, kernel_size=3, padding=1)
    self._b3 = torch.nn.BatchNorm2d(self._c3.out_channels)
    self._c4 = torch.nn.Conv2d(self._c3.out_channels, 128, kernel_size=3, padding=1)
    self._b4 = torch.nn.BatchNorm2d(self._c4.out_channels)
    self._m2 = torch.nn.MaxPool2d(2)
    self._d2 = torch.nn.Dropout(p=0.25)
    self._d3 = torch.nn.Dropout(p=0.25)
    if cifar100: # CIFAR-100
        self._f1 = torch.nn.Linear(8192, 256)
        self._f2 = torch.nn.Linear(self._f1.out_features, 100)
    else: # CIFAR-10
        self._f1 = torch.nn.Linear(8192, 128)
        self._f2 = torch.nn.Linear(self._f1.out_features, 10)

  def forward(self, x):
    """ Model's forward pass.
    Args:
      x Input tensor
    Returns:
      Output tensor
    """
    activation = torch.nn.functional.relu
    flatten    = lambda x: x.view(x.shape[0], -1)
    logsoftmax = torch.nn.functional.log_softmax
    # Forward pass
    x = self._c1(x)
    x = activation(x)
    x = self._b1(x)
    x = self._c2(x)
    x = activation(x)
    x = self._b2(x)
    x = self._m1(x)
    x = self._d1(x)
    x = self._c3(x)
    x = activation(x)
    x = self._b3(x)
    x = self._c4(x)
    x = activation(x)
    x = self._b4(x)
    x = self._m2(x)
    x = self._d2(x)
    x = flatten(x)
    x = self._f1(x)
    x = activation(x)
    x = self._d3(x)
    x = self._f2(x)
    x = logsoftmax(x, dim=1)
    return x

def cnn(*args, **kwargs):
  """ Build a new simple, convolutional model.
  Args:
    ... Forwarded (keyword-)arguments
  Returns:
    Convolutional model
  """
  global _CNN
  return _CNN(*args, **kwargs)

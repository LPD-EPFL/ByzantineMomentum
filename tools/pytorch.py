# coding: utf-8
###
 # @file   pytorch.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2020 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Helpers relative to PyTorch.
###

__all__ = ["relink", "flatten", "grad_of", "grads_of", "pnm"]

import torch
import types

import tools

# ---------------------------------------------------------------------------- #
# "Flatten" and "relink" operations

def relink(tensors, common):
  """ "Relink" the tensors of class (deriving from) Tensor by making them point to another contiguous segment of memory.
  Args:
    tensors Generator of/iterable on instances of/deriving from Tensor, all with the same dtype
    common  Flat tensor of sufficient size to use as underlying storage, with the same dtype as the given tensors
  Returns:
    Given common tensor
  """
  # Convert to tuple if generator
  if isinstance(tensors, types.GeneratorType):
    tensors = tuple(tensors)
  # Relink each given tensor to its segment on the common one
  pos = 0
  for tensor in tensors:
    npos = pos + tensor.numel()
    tensor.data = common[pos:npos].view(*tensor.shape)
    pos = npos
  # Finalize and return
  common.linked_tensors = tensors
  return common

def flatten(tensors):
  """ "Flatten" the tensors of class (deriving from) Tensor so that they all use the same contiguous segment of memory.
  Args:
    tensors Generator of/iterable on instances of/deriving from Tensor, all with the same dtype
  Returns:
    Flat tensor (with the same dtype as the given tensors) that contains the memory used by all the given Tensor (or derived instances), in emitted order
  """
  # Convert to tuple if generator
  if isinstance(tensors, types.GeneratorType):
    tensors = tuple(tensors)
  # Common tensor instantiation and reuse
  common = torch.cat(tuple(tensor.view(-1) for tensor in tensors))
  # Return common tensor
  return relink(tensors, common)

# ---------------------------------------------------------------------------- #
# Gradient access

def grad_of(tensor):
  """ Get the gradient of a given tensor, make it zero if missing.
  Args:
    tensor Given instance of/deriving from Tensor
  Returns:
    Gradient for the given tensor
  """
  # Get the current gradient
  grad = tensor.grad
  if grad is not None:
    return grad
  # Make and set a zero-gradient
  grad = torch.zeros_like(tensor)
  tensor.grad = grad
  return grad

def grads_of(tensors):
  """ Iterate of the gradients of the given tensors, make zero gradients if missing.
  Args:
    tensors Generator of/iterable on instances of/deriving from Tensor
  Returns:
    Generator of the gradients of the given tensors, in emitted order
  """
  return (grad_of(tensor) for tensor in tensors)

# ---------------------------------------------------------------------------- #
# Save image as PGM/PBM stream

def pnm(fd, tn):
  """ Save a 2D/3D tensor as a PGM/PBM stream.
  Args:
    fd File descriptor opened for writing binary streams
    tn A 2D/3D tensor to convert and save
  Notes:
    The input tensor is "intelligently" squeezed before processing
    For 2D tensor, assuming black is 1. and white is 0. (clamp between [0, 1])
    For 3D tensor, the first dimension must be the 3 color channels RGB (all between [0, 1])
  """
  shape = tuple(tn.shape)
  # Intelligent squeezing
  while len(tn.shape) > 3 and tn.shape[0] == 1:
    tn = tn[0]
  # Colored image generation
  if len(tn.shape) == 3:
    if tn.shape[0] == 1:
      tn = tn[0]
      # And continue on gray-scale
    elif tn.shape[0] != 3:
      raise tools.UserException("Expected 3 color channels for the first dimension of a 3D tensor, got %d channels" % tn.shape[0])
    else:
      fd.write(("P6\n%d %d 255\n" % tn.shape[1:]).encode())
      fd.write(bytes(tn.transpose(0, 2).transpose(0, 1).mul(256).clamp_(0., 255.).byte().storage()))
      return
  # Gray-scale image generation
  if len(tn.shape) == 2:
    fd.write(("P5\n%d %d 255\n" % tn.shape).encode())
    fd.write(bytes((1.0 - tn).mul_(256).clamp_(0., 255.).byte().storage()))
    return
  # Invalid tensor shape
  raise tools.UserException("Expected a 2D or 3D tensor, got %d dimensions %r" % (len(shape), tuple(shape)))

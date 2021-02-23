# coding: utf-8
###
 # @file   pytorch.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Helpers relative to PyTorch.
###

__all__ = ["relink", "flatten", "grad_of", "grads_of", "compute_avg_dev_max",
           "AccumulatedTimedContext", "weighted_mse_loss", "WeightedMSELoss",
           "regression", "pnm"]

import math
import time
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
# Useful computations

def compute_avg_dev_max(samples):
  """ Compute the norm average and norm standard deviation of gradient samples.
  Args:
    samples Given gradient samples
  Returns:
    Computed average gradient (None if no sample), norm average, norm standard deviation, average maximum absolute coordinate
  """
  # Trivial case no sample
  if len(samples) == 0:
    return None, math.nan, math.nan, math.nan
  # Compute average gradient and max abs coordinate
  grad_avg = samples[0].clone().detach_()
  for grad in samples[1:]:
    grad_avg.add_(grad)
  grad_avg.div_(len(samples))
  norm_avg = grad_avg.norm().item()
  norm_max = grad_avg.abs().max().item()
  # Compute norm standard deviation
  if len(samples) >= 2:
    norm_var = 0.
    for grad in samples:
      grad = grad.sub(grad_avg)
      norm_var += grad.dot(grad).item()
    norm_var /= len(samples) - 1
    norm_dev = math.sqrt(norm_var)
  else:
    norm_dev = math.nan
  # Return norm average and deviation
  return grad_avg, norm_avg, norm_dev, norm_max

# ---------------------------------------------------------------------------- #
# Simple timing context

class AccumulatedTimedContext:
  """ Accumulated timed context class, that do not print.
  """

  def _sync_cuda(self):
    """ Synchronize CUDA streams (if requested and relevant).
    """
    if self._sync and torch.cuda.is_available():
      torch.cuda.synchronize()

  def __init__(self, initial=0., *, sync=False):
    """ Zero runtime constructor.
    Args:
      initial Initial total runtime (in s)
      sync    Whether to synchronize with already running/launched CUDA streams
    """
    # Finalization
    self._total = initial  # Total runtime (in s)
    self._sync  = sync

  def __enter__(self):
    """ Enter context: start chrono.
    Returns:
      Self
    """
    # Synchronize CUDA streams (if requested and relevant)
    self._sync_cuda()
    # "Start" chronometer
    self._chrono = time.time()
    # Return self
    return self

  def __exit__(self, *args, **kwargs):
    """ Exit context: stop chrono and accumulate elapsed time.
    Args:
      ... Ignored
    """
    # Synchronize CUDA streams (if requested and relevant)
    self._sync_cuda()
    # Accumulate elapsed time (in s)
    self._total += time.time() - self._chrono

  def __str__(self):
    """ Pretty-print total runtime.
    Returns:
      Total runtime string with unit
    """
    # Get total runtime (in ns)
    runtime = self._total * 1000000000.
    # Recover ideal unit
    for unit in ("ns", "µs", "ms"):
      if runtime < 1000.:
        break
      runtime /= 1000.
    else:
      unit = "s"
    # Format and return string
    return f"{runtime:.3g} {unit}"

  def current_runtime(self):
    """ Get the current accumulated runtime.
    Returns:
      Current runtime (in s)
    """
    return self._total

# ---------------------------------------------------------------------------- #
# Regression helper

def weighted_mse_loss(tno, tne, tnw):
  """ Weighted mean square error loss.
  Args:
    tno Output tensor
    tne Expected output tensor
    tnw Weight tensor
  Returns:
    Associated loss tensor
  """
  return torch.mean((tno - tne).pow_(2).mul_(tnw))

class WeightedMSELoss(torch.nn.Module):
  """ Weighted mean square error loss class.
  """

  def __init__(self, weight, *args, **kwargs):
    """ Weight binding constructor.
    Args:
      weight Weight to bind
      ...    Forwarding (keyword-)arguments
    """
    super().__init__(*args, **kwargs)
    self.register_buffer("weight", weight)

  def forward(self, tno, tne):
    """ Compute the weighted mean square error.
    Args:
      tno Output tensor
      tne Expeced output tensor
    Returns:
      Associated loss tensor
    """
    return weighted_mse_loss(tno, tne, self.weight)

def regression(func, vars, data, loss=torch.nn.MSELoss(), opt=torch.optim.Adam, steps=1000):
  """ Performs a regression (mere optimization of variables) for the given function.
  Args:
    func  Function to fit
    vars  Iterable of the free tensor variables to optimize
    data  Tuple of (input data tensor, expected output data tensor)
    loss  Loss function to use, taking (output, expected output)
    opt   Optimizer to use (function mapping a once-iterable of tensors to an optimizer instance)
    steps Number of optimization epochs to perform (1 epoch/step)
  Returns:
    Step at which optimization stopped
  """
  # Prepare
  tni = data[0]
  tne = data[1]
  opt = opt(vars)
  # Optimize
  for step in range(steps):
    with torch.enable_grad():
      opt.zero_grad()
      res = loss(func(tni), tne)
      if torch.isnan(res).any().item():
        return step
      res.backward()
      opt.step()
  return steps

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
      raise tools.UserException(f"Expected 3 color channels for the first dimension of a 3D tensor, got {tn.shape[0]} channels")
    else:
      fd.write((f"P6\n{tn.shape[1]} {tn.shape[2]} 255\n").encode())
      fd.write(bytes(tn.transpose(0, 2).transpose(0, 1).mul(256).clamp_(0., 255.).byte().storage()))
      return
  # Gray-scale image generation
  if len(tn.shape) == 2:
    fd.write((f"P5\n{tn.shape[0]} {tn.shape[1]} 255\n").encode())
    fd.write(bytes((1.0 - tn).mul_(256).clamp_(0., 255.).byte().storage()))
    return
  # Invalid tensor shape
  raise tools.UserException(f"Expected a 2D or 3D tensor, got {len(shape)} dimensions {tuple(shape)!r}")

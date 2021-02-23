# coding: utf-8
###
 # @file   loss.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Loss/criterion wrappers/helpers.
###

__all__ = ["Loss", "Criterion"]

import tools

import torch

# ---------------------------------------------------------------------------- #
# Loss (derivable)/criterion (non-derivable) wrapper classes

class Loss:
  """ Loss (must be derivable) wrapper class.
  """

  __reserved_init = object()

  @staticmethod
  def _l1loss(output, target, params):
    """ l1 loss implementation
    Args:
      ...    Ignored arguments
      params Flat parameter tensor
    Returns:
      l1 loss
    """
    return params.norm(p=1)

  @staticmethod
  def _l2loss(output, target, params):
    """ l2 loss implementation
    Args:
      ...    Ignored arguments
      params Flat parameter tensor
    Returns:
      l2 loss
    """
    return params.norm()

  @classmethod
  def _l1loss_builder(self):
    """ l1 loss builder.
    Returns:
      L1 loss instance
    """
    return self(self.__reserved_init, self._l1loss, None, "l1")

  @classmethod
  def _l2loss_builder(self):
    """ l2 loss builder.
    Returns:
      L2 loss instance
    """
    return self(self.__reserved_init, self._l2loss, None, "l2")

  # Map 'lower-case names' -> 'loss constructor' available in PyTorch
  __losses = None

  @staticmethod
  def _make_drop_params(builder):
    """ Make a builder that will wrap the built function so to drop the 'params' parameter.
    Args:
      builder Builder function to wrap
    Returns:
      Wrapped builder function
    """
    def drop_builder(*args, **kwargs):
      loss = builder(*args, **kwargs)
      def drop_loss(output, target, params):
        return loss(output, target)
      return drop_loss
    return drop_builder

  @classmethod
  def _get_losses(self):
    """ Lazy-initialize and return the map '__losses'.
    Returns:
      '__losses'
    """
    # Fast-path already loaded
    if self.__losses is not None:
      return self.__losses
    # Initialize the dictionary
    self.__losses = dict()
    # Simply populate this dictionary
    for name in dir(torch.nn.modules.loss):
      if len(name) < 5 or name[0] == "_" or name[-4:] != "Loss": # Heuristically ignore non-loss members
        continue
      builder = getattr(torch.nn.modules.loss, name)
      if isinstance(builder, type): # Still an heuristic
        self.__losses[name[:-4].lower()] = self._make_drop_params(builder)
    # Add/replace the l1 and l2 losses
    self.__losses["l1"] = self._l1loss_builder
    self.__losses["l2"] = self._l2loss_builder
    # Return the dictionary
    return self.__losses

  def __init__(self, name_build, *args, **kwargs):
    """ Loss constructor.
    Args:
      name_build Loss name or constructor function
      ...        Additional (keyword-)arguments forwarded to the constructor
    """
    # Reserved custom initialization
    if name_build is type(self).__reserved_init:
      self._loss = args[0]
      self._fact = args[1]
      self._name = args[2]
      return
    # Recover name/constructor
    if callable(name_build):
      name  = tools.fullqual(name_build)
      build = name_build
    else:
      losses = type(self)._get_losses()
      name   = str(name_build)
      build  = losses.get(name, None)
      if build is None:
        raise tools.UnavailableException(losses, name, what="loss name")
    # Build loss
    loss = build(*args, **kwargs)
    # Finalization
    self._loss = loss
    self._fact = None
    self._name = name

  def _str_make(self):
    """ Make the formatted part of the nicely printable string representation of this loss.
    Returns:
      Formatted part
    """
    return self._name if self._fact is None else f"{self._fact} × {self._name}"

  def __str__(self):
    """ Compute the "informal", nicely printable string representation of this loss.
    Returns:
      Nicely printable string
    """
    return f"loss {self._str_make()}"

  def __call__(self, output, target, params):
    """ Compute the loss from the output and the target.
    Args:
      output Output tensor from the model
      target Expected tensor
      params Parameter vector
    Returns:
      Computed loss tensor
    """
    res = self._loss(output, target, params)
    if self._fact is not None:
      res *= self._fact
    return res

  def __add__(self, loss):
    """ Add the current loss to the given loss.
    Args:
      loss Given loss
    Returns:
      Sum of the two losses
    """
    def add(output, target, params):
      return self(output, target, params) + loss(output, target, params)
    return type(self)(type(self).__reserved_init, add, None, f"({self._str_make()} + {loss._str_make()})")

  def __mul__(self, factor):
    """ Multiply the current loss by a given factor.
    Args:
      factor Given factor
    Returns:
      New loss, factor of the current loss
    """
    def mul(output, target, params):
      return self(output, target, params) * factor
    return type(self)(type(self).__reserved_init, mul, factor * (1. if self._fact is None else self._fact), self._name)

  def __rmul__(self, *args, **kwargs):
    """ Forward the call to '__mul__'.
    Args:
      ... Forwarded (keyword-)arguments
    Returns:
      Forwarded return value
    """
    return self.__mul__(*args, **kwargs)

  def __imul__(self, factor):
    """ In-place multiply the current loss by a given factor.
    Args:
      factor Given factor
    Returns:
      Current loss
    """
    self._fact = factor * (1. if self._fact is None else self._fact)
    return self

class Criterion:
  """ Criterion (1D tensor [#correct classification, batch size]) wrapper class.
  """

  class _TopkCriterion:
    """ Top-k criterion helper class.
    """

    def __init__(self, k=1):
      """ Value of 'k' constructor.
      Args:
        k Value of 'k' to use
      """
      # Finalization
      self.k = k

    def __call__(self, output, target):
      """ Compute the criterion from the output and the target.
      Args:
        output Batch × model logits
        target Batch × target index
      Returns:
        1D-tensor [#correct classification, batch size]
      """
      res = (output.topk(self.k, dim=1)[1] == target.view(-1).unsqueeze(1)).any(dim=1).sum()
      return torch.cat((res.unsqueeze(0), torch.tensor(target.shape[0], dtype=res.dtype, device=res.device).unsqueeze(0)))

  class _SigmoidCriterion:
    """ Sigmoid criterion helper class.
    """

    def __call__(self, output, target):
      """ Compute the criterion from the output and the target.
      Args:
        output Batch × model logits (expected in [0, 1])
        target Batch × target index (expected in {0, 1})
      Returns:
        1D-tensor [#correct classification, batch size]
      """
      correct = target.sub(output).abs_() < 0.5
      res = torch.empty(2, dtype=output.dtype, device=output.device)
      res[0] = correct.sum()
      res[1] = len(correct)
      return res

  # Map 'lower-case names' -> 'loss constructor' available in PyTorch
  __criterions = None

  @classmethod
  def _get_criterions(self):
    """ Lazy-initialize and return the map '__criterions'.
    Returns:
      '__criterions'
    """
    # Fast-path already loaded
    if self.__criterions is not None:
      return self.__criterions
    # Initialize the dictionary
    self.__criterions = {
      "top-k": self._TopkCriterion,
      "sigmoid": self._SigmoidCriterion }
    # Return the dictionary
    return self.__criterions

  def __init__(self, name_build, *args, **kwargs):
    """ Criterion constructor.
    Args:
      name_build Criterion name or constructor function
      ...        Additional (keyword-)arguments forwarded to the constructor
    """
    # Recover name/constructor
    if callable(name_build):
      name  = tools.fullqual(name_build)
      build = name_build
    else:
      crits = type(self)._get_criterions()
      name  = str(name_build)
      build = crits.get(name, None)
      if build is None:
        raise tools.UnavailableException(crits, name, what="criterion name")
    # Build criterion
    crit = build(*args, **kwargs)
    # Finalization
    self._crit = crit
    self._name = name

  def __str__(self):
    """ Compute the "informal", nicely printable string representation of this criterion.
    Returns:
      Nicely printable string
    """
    return f"criterion {self._name}"

  def __call__(self, output, target):
    """ Compute the criterion from the output and the target.
    Args:
      output Output tensor from the model
      target Expected tensor
    Returns:
      Computed criterion tensor
    """
    return self._crit(output, target)

# coding: utf-8
###
 # @file   optimizer.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Optimizer wrapper.
###

__all__ = ["Optimizer"]

import tools

import torch

# ---------------------------------------------------------------------------- #
# Optimizer wrapper class

class Optimizer:
  """ Optimizer wrapper class.
  """

  # Map 'lower-case names' -> 'optimizer constructor' available in PyTorch
  __optimizers = None

  @classmethod
  def _get_optimizers(self):
    """ Lazy-initialize and return the map '__optimizers'.
    Returns:
      '__optimizers'
    """
    # Fast-path already loaded
    if self.__optimizers is not None:
      return self.__optimizers
    # Initialize the dictionary
    self.__optimizers = dict()
    # Simply populate this dictionary
    for name in dir(torch.optim):
      if len(name) == 0 or name[0] == "_": # Ignore "protected" members
        continue
      builder = getattr(torch.optim, name)
      if isinstance(builder, type) and builder is not torch.optim.Optimizer and issubclass(builder, torch.optim.Optimizer):
        self.__optimizers[name.lower()] = builder
    # Return the dictionary
    return self.__optimizers

  def __init__(self, name_build, model, *args, **kwargs):
    """ Optimizer constructor.
    Args:
      name_build Optimizer name or constructor function
      model      Model to optimize
      ...        Additional (keyword-)arguments forwarded to the constructor
    """
    # Recover name/constructor
    if callable(name_build):
      name  = tools.fullqual(name_build)
      build = name_build
    else:
      optims = type(self)._get_optimizers()
      name   = str(name_build)
      build  = optims.get(name, None)
      if build is None:
        raise tools.UnavailableException(optims, name, what="optimizer name")
    # Build optimizer
    optim = build(model._model.parameters(), *args, **kwargs)
    # Finalization
    self._optim = optim
    self._name  = name

  def __getattr__(self, *args):
    """ Get attribute on the optimizer instance.
    Args:
      name    Name of the attribute to get
      default Default value returned if the attribute does not exist
    Returns:
      Forwarded attribute
    """
    if len(args) == 1:
      return getattr(self._optim, args[0])
    if len(args) == 2:
      return getattr(self._optim, args[0], args[1])
    raise RuntimeError("'Optimizer.__getattr__' called with the wrong number of parameters")

  def __str__(self):
    """ Compute the "informal", nicely printable string representation of this optimizer.
    Returns:
      Nicely printable string
    """
    return f"optimizer {self._name}"

  def set_lr(self, lr):
    """ Set the learning rate of the optimizer
    Args:
      lr Learning rate to set (for each parameter group)
    """
    for pg in self._optim.param_groups:
      pg["lr"] = lr

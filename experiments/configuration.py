# coding: utf-8
###
 # @file   configuration.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Configuration wrapper.
###

__all__ = ["Configuration"]

import tools

from   collections.abc import Mapping
import torch

# ---------------------------------------------------------------------------- #
# Trivial tensor configuration holder (dtype, device, ...) class

class Configuration(Mapping):
  """ Immutable tensor configuration holder class.
  """

  # Default selected device (GPU if available, else CPU)
  default_device = "cuda" if torch.cuda.is_available() else "cpu"

  def __init__(self, device=None, dtype=None, noblock=False, relink=False):
    """ Immutable initialization constructor.
    Args:
      device  Device (either instance, formatted name or None) to use
      dtype   Datatype to use, None for PyTorch default
      noblock To try and avoid using blocking memory transfer operations from the host
      relink  Relink instead of copying by default in some assignment operations
    """
    # Convert formatted device name to device instance
    if device is None:
      # Use default device
      device = type(self).default_device
    if isinstance(device, str):
      # Warn if CUDA is requested but not available
      if not torch.cuda.is_available() and device[:4] == "cuda":
        device = "cpu"
        tools.warning("CUDA is unavailable on this node, falling back to CPU in the configuration", context="experiments")
      # Convert
      device = torch.device(device)
    # Resolve the current default dtype if unspecified
    if dtype is None:
      dtype = torch.get_default_dtype()
    # Finalization
    self._args = {
      "device": device,
      "dtype": dtype,
      "non_blocking": noblock }
    self.relink = relink

  def __len__(self):
    """ Return the number of contained configuration entries.
    Returns:
      Number of configuration entries
    """
    return len(self._args)

  def __getitem__(self, name):
    """ Get a configuration value from its name.
    Args:
      name Configuration name
    Returns:
      Associated configuration value
    """
    return self._args[name]

  def __iter__(self):
    """ Build an iterator over all the configuration entries.
    Return:
      Built iterator
    """
    return self._args.__iter__()

  def __str__(self):
    """ Compute the "informal", nicely printable string representation of this configuration.
    Returns:
      Nicely printable string
    """
    temp = self._args.copy()
    temp["relink"] = self.relink
    return str(temp)

  def __repr__(self):
    """ Compute the "official", Python-code string representation of this configuration.
    Returns:
      Python-code string evaluating (under conditions) to this configuration
    """
    display = {"non_blocking": "noblock"}
    argrepr = (", ").join(f"{display.get(key, key)}={val!r}" for key, val in self._args.items())
    return f"Configuration({argrepr}, relink={self.relink})"

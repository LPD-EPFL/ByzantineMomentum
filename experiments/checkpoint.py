# coding: utf-8
###
 # @file   checkpoint.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Checkpoint helpers.
###

__all__ = ["Checkpoint", "Storage"]

import tools

import copy
import pathlib
import torch

from .model import Model
from .optimizer import Optimizer

# ---------------------------------------------------------------------------- #
# Checkpoint helper class

class Checkpoint:
  """ A collection of state dictionaries with saving/loading helpers.
  """

  # Transfer for handling local package's classes
  _transfers = {
    Model: (lambda x: x._model),
    Optimizer: (lambda x: x._optim) }

  @classmethod
  def _prepare(self, instance):
    """ Prepare the given instance for checkpointing.
    Args:
      instance Instance to snapshot/restore
    Returns:
      Checkpoint-able instance, key for the associated storage
    """
    # Recover instance's class
    cls = type(instance)
    # Transfer if available
    if cls in self._transfers:
      res = self._transfers[cls](instance)
    else:
      res = instance
    # Assert the instance is checkpoint-able
    for prop in ("state_dict", "load_state_dict"):
      if not callable(getattr(res, prop, None)):
        raise tools.UserException(f"Given instance {instance!r} is not checkpoint-able (missing callable member {prop!r})")
    # Return the instance and the associated storage key
    return res, tools.fullqual(cls)

  def __init__(self):
    """ Empty checkpoint constructor.
    """
    # Finalization
    self._store = dict()
    if __debug__:
      self._copied = dict() # Booleans for tracking possible bugs, 'key in _store' <=> 'key in _copied'

  def snapshot(self, instance, overwrite=False, deepcopy=False, nowarnref=False):
    """ Take/overwrite the snapshot for a given instance.
    Args:
      instance  Instance to snapshot
      overwrite Overwrite any existing snapshot for the same class
      deepcopy  Deep copy instance's state dictionary instead of referencing
      nowarnref To always avoid a warning in debug mode if restoring a state dictionary reference is the wanted behavior
    Returns:
      self
    """
    instance, key = type(self)._prepare(instance)
    # Snapshot the state dictionary
    if not overwrite and key in self._store:
      raise tools.UserException(f"A snapshot for {key!r} is already stored in the checkpoint")
    if deepcopy:
      self._store[key] = copy.deepcopy(instance.state_dict())
    else:
      self._store[key] = instance.state_dict().copy()
    # Track whether a deepcopy was made (or whether restoring a reference is the expected behavior)
    if __debug__:
      self._copied[key] = deepcopy or nowarnref
    # Enable chaining
    return self

  def restore(self, instance, nothrow=False):
    """ Restore the snapshot for a given instance, warn if restoring a reference.
    Args:
      instance Instance to restore
      nothrow  Do not raise exception if no snapshot available for the instance
    Returns:
      self
    """
    instance, key = type(self)._prepare(instance)
    # Restore the state dictionary
    if key in self._store:
      instance.load_state_dict(self._store[key])
      # Check if restoring a reference
      if __debug__ and not self._copied[key]:
        tools.warning(f"Restoring a state dictionary reference in an instance of {tools.fullqual(type(instance))}; the resulting behavior may not be the one expected")
    elif not nothrow:
      raise tools.UserException(f"No snapshot for {key!r} is available in the checkpoint")
    # Enable chaining
    return self

  def load(self, filepath, overwrite=False):
    """ Load/overwrite the storage from the given file.
    Args:
      filepath  Given file path
      overwrite Allow to overwrite any stored snapshot
    Returns:
      self
    """
    # Check if empty
    if not overwrite and len(self._store) > 0:
      raise tools.UserException("Unable to load into a non-empty checkpoint")
    # Load the file
    self._store = torch.load(filepath)
    # Reset the 'copied' flags accordingly
    if __debug__:
      self._copied.clear()
      for key in self._store.keys():
        self._copied[key] = True
    # Enable chaining
    return self

  def save(self, filepath, overwrite=False):
    """ Save the current checkpoint in the given file.
    Args:
      filepath  Given file path
      overwrite Allow to overwrite if the file already exists
    Returns:
      self
    """
    # Check if file already exists
    if pathlib.Path(filepath).exists() and not overwrite:
      raise tools.UserException(f"Unable to save checkpoint in existing file {str(filepath)!r} (overwriting has not been allowed by the caller)")
    # (Over)write the file
    torch.save(self._store, filepath)
    # Enable chaining
    return self

# ---------------------------------------------------------------------------- #
# Dictionary that implements "state_dict protocol"

class Storage(dict):
  """ Dictionary that implements "state_dict protocol" class.
  """

  def state_dict(self):
    """ Access the state dictionary.
    Returns:
      self
    """
    return self

  def load_state_dict(self, state):
    """ Update the state dictionary.
    Args:
      state State to update the current storage with
    """
    self.update(state)

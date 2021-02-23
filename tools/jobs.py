# coding: utf-8
###
 # @file   jobs.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2020-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Simple job management for reproduction scripts.
###

__all__ = ["dict_to_cmdlist", "Command", "Jobs"]

import shlex
import subprocess
import threading

import tools

# ---------------------------------------------------------------------------- #
# Helpers

def move_directory(path):
  """ Move existing directory to a new location (with a numbering scheme).
  Args:
    path Path to the directory to create
  Returns:
    'path' (to enable chaining)
  """
  # Move directory if it exists
  if path.exists():
    if not path.is_dir():
      raise RuntimeError(f"Expected to find nothing or (a symlink to) a directory at {str(path)!r}")
    i = 0
    while True:
      mvpath = path.parent / f"{path.name}.{i}"
      if not mvpath.exists():
        path.rename(mvpath)
        break
      i += 1
  # Enable chaining
  return path

def dict_to_cmdlist(dp):
  """ Transform a dictionary into a list of command arguments.
  Args:
    dp Dictionary mapping parameter name (to prepend with "--") to parameter value (to convert to string)
  Returns:
    Associated list of command arguments
  Notes:
    For entries mapping to 'bool', the parameter is included/discarded depending on whether the value is True/False
    For entries mapping to 'list' or 'tuple', the parameter is followed by all the values as strings
  """
  cmd = list()
  for name, value in dp.items():
    if isinstance(value, bool):
      if value:
        cmd.append(f"--{name}")
    else:
      if any(isinstance(value, typ) for typ in (list, tuple)):
        cmd.append(f"--{name}")
        for subval in value:
          cmd.append(str(subval))
      elif value is not None:
        cmd.append(f"--{name}")
        cmd.append(str(value))
  return cmd

# ---------------------------------------------------------------------------- #
# Job command class

class Command:
  """ Simple job command class, that builds a command from a dictionary of parameters.
  """

  def __init__(self, command):
    """ Bind constructor.
    Args:
      command Command iterable (will be copied)
    """
    self._basecmd = list(command)

  def build(self, seed, device, resdir):
    """ Build the final command line.
    Args:
      seed   Seed to use
      device Device to use
      resdir Target directory path
    Returns:
      Final command list
    """
    # Build final command list
    cmd = self._basecmd.copy()
    for name, value in (("seed", seed), ("device", device), ("result-directory", resdir)):
      cmd.append(f"--{name}")
      cmd.append(shlex.quote(value if isinstance(value, str) else str(value)))
    # Return final command list
    return cmd

# ---------------------------------------------------------------------------- #
# Job class

class Jobs:
  """ Take experiments to run and runs them on the available devices, managing repetitions.
  """

  @staticmethod
  def _run(topdir, name, seed, device, command):
    """ Run the attack experiments with the given named parameters.
    Args:
      topdir  Parent result directory
      name    Experiment unique name
      seed    Experiment seed
      device  Device on which to run the experiments
      command Command to run
    """
    # Add seed to name
    name = f"{name}-{seed}"
    # Process experiment
    with tools.Context(name, "info"):
      finaldir = topdir / name
      # Check whether the experiment was already successful
      if finaldir.exists():
        tools.info("Experiment already processed.")
        return
      # Move-make the pending result directory
      resdir = move_directory(topdir / f"{name}.pending")
      resdir.mkdir(mode=0o755, parents=True)
      # Build the command
      args = command.build(seed, device, resdir)
      # Launch the experiment and write the standard output/error
      tools.trace((" ").join(shlex.quote(arg) for arg in args))
      cmd_res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      if cmd_res.returncode == 0:
        tools.info("Experiment successful")
      else:
        tools.warning("Experiment failed")
        finaldir = topdir / f"{name}.failed"
        move_directory(finaldir)
      resdir.rename(finaldir)
      (finaldir / "stdout.log").write_bytes(cmd_res.stdout)
      (finaldir / "stderr.log").write_bytes(cmd_res.stderr)

  def _worker_entrypoint(self, device):
    """ Worker entry point.
    Args:
      device Device to use
    """
    while True:
      # Take a pending experiment, or exit if requested
      with self._lock:
        while True:
          # Check if must exit
          if self._jobs is None:
            return
          # Check and pick the first pending experiment, if available
          if len(self._jobs) > 0:
            name, seed, command = self._jobs.pop()
            break
          # Wait for new job notification
          self._cvready.wait()
      # Run the picked experiment
      self._run(self._res_dir, name, seed, device, command)

  def __init__(self, res_dir, devices=["cpu"], devmult=1, seeds=tuple(range(1, 6))):
    """ Initialize the instance, launch the worker pool.
    Args:
      res_dir Path to the directory containing the result sub-directories
      devices List/tuple of the devices to use in parallel
      devmult How many experiments are run in parallel per device
      seeds   List/tuple of seeds to repeat the experiments with
    """
    # Initialize instance
    self._res_dir = res_dir
    self._jobs    = list() # List of tuples (name, seed, command), or None to signal termination
    self._workers = list() # Worker pool, one per target device
    self._devices = devices
    self._seeds   = seeds
    self._lock    = threading.Lock()
    self._cvready = threading.Condition(lock=self._lock) # Signal jobs have been added and must be processed, or the worker must quit
    self._cvdone  = threading.Condition(lock=self._lock) # Signal jobs have all been processed
    # Launch the worker pool
    for _ in range(devmult):
      for device in devices:
        thread = threading.Thread(target=self._worker_entrypoint, name=device, args=(device,))
        thread.start()
        self._workers.append(thread)

  def get_seeds(self):
    """ Get the list of seeds used for repeating the experiments.
    Returns:
      List/tuple of seeds used
    """
    return self._seeds

  def close(self):
    """ Close and wait for the worker pool, discarding not yet started submission.
    """
    # Close the manager
    with self._lock:
      # Check if already closed
      if self._jobs is None:
        return
      # Reset submission list
      self._jobs = None
      # Notify all the workers
      self._cvready.notify_all()
    # Wait for all the workers
    for worker in self._workers:
      worker.join()

  def submit(self, name, command):
    """ Submit an experiment to be run with each seed on any available device.
    Args:
      name    Experiment unique name
      command Command to process
    """
    with self._lock:
      # Check if not closed
      if self._jobs is None:
        raise RuntimeError("Experiment manager cannot take new jobs as it has been closed")
      # Submit the experiment with each seed
      for seed in self._seeds:
        self._jobs.insert(0, (name, seed, command))
      self._cvready.notify(n=len(self._seeds))

  def wait(self, predicate=None):
    """ Wait for all the submitted jobs to be processed.
    Args:
      predicate Custom predicate to call to check whether must stop waiting
    """
    while True:
      with self._lock:
        # Wait for condition or timeout
        self._cvdone.wait(timeout=1.)
        # Check status
        if self._jobs is None:
          break
        if len(self._jobs) == 0:
          break
        if not any(worker.is_alive() for worker in self._workers):
          break
        if predicate is not None and predicate():
          break

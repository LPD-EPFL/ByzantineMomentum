# coding: utf-8
###
 # @file   reproduce.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2020 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Reproduce the (missing) experiments and plots.
###

import tools
tools.success("Module loading...")

import argparse
import pathlib
import shlex
import signal
import subprocess
import sys
import threading

import torch

import experiments

# ---------------------------------------------------------------------------- #
# Miscellaneous initializations
tools.success("Miscellaneous initializations...")

# "Exit requested" global variable accessors
exit_is_requested, exit_set_requested = tools.onetime("exit")

# Signal handlers
signal.signal(signal.SIGINT, exit_set_requested)
signal.signal(signal.SIGTERM, exit_set_requested)

# ---------------------------------------------------------------------------- #
# Command-line processing
tools.success("Command-line processing...")

def process_commandline():
  """ Parse the command-line and perform checks.
  Returns:
    Parsed configuration
  """
  # Description
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument("--data-directory",
    type=str,
    default="results-data",
    help="Path of the data directory, containing the data gathered from the experiments")
  parser.add_argument("--plot-directory",
    type=str,
    default="results-plot",
    help="Path of the plot directory, containing the graphs traced from the experiments")
  parser.add_argument("--devices",
    type=str,
    default="auto",
    help="Comma-separated list of devices on which to run the experiments, used in a round-robin fashion")
  parser.add_argument("--supercharge",
    type=int,
    default=1,
    help="How many experiments are run in parallel per device, must be positive")
  # Parse command line
  return parser.parse_args(sys.argv[1:])

with tools.Context("cmdline", "info"):
  args = process_commandline()
  # Check the "supercharge" parameter
  if args.supercharge < 1:
    tools.fatal("Expected a positive supercharge value, got %d" % args.supercharge)
  # Make the result directories
  def check_make_dir(path):
    path = pathlib.Path(path)
    if path.exists():
      if not path.is_dir():
        tools.fatal("Given path %r must point to a directory" % (str(path),))
    else:
      path.mkdir(mode=0o755, parents=True)
    return path
  args.data_directory = check_make_dir(args.data_directory)
  args.plot_directory = check_make_dir(args.plot_directory)
  # Preprocess/resolve the devices to use
  if args.devices == "auto":
    if torch.cuda.is_available():
      args.devices = list("cuda:%d" % i for i in range(torch.cuda.device_count()))
    else:
      args.devices = ["cpu"]
  else:
    args.devices = list(name.strip() for name in args.devices.split(","))

# ---------------------------------------------------------------------------- #
# Serial preloading of the dataset
tools.success("Pre-downloading datasets...")

# Pre-load the datasets to prevent the first parallel runs from downloading them several times
with tools.Context("dataset", "info"):
  for name in ("mnist", "cifar10"):
    with tools.Context(name, "info"):
      experiments.make_datasets(name, 1, 1)

# ---------------------------------------------------------------------------- #
# Run (missing) experiments
tools.success("Running experiments...")

class Jobs:
  """ Take experiments to run and runs them on the available devices, managing repetitions.
  """

  @staticmethod
  def _run(name, seed, device, params):
    """ Run the attack experiments with the given named parameters.
    Args:
      name   Experiment unique name
      seed   Experiment seed
      device Device on which to run the experiments
      params Named parameters
    """
    # Add seed to name
    name = "%s-%d" % (name, seed)
    # Process experiment
    with tools.Context(name, "info"):
      # Build and set the result directory
      result_dir = args.data_directory / name
      if result_dir.exists():
        tools.info("Experiment already processed.")
        return
      result_dir.mkdir(mode=0o755, parents=True)
      # Add the missing options
      params["seed"] = str(seed)
      params["device"] = device
      params["result-directory"] = str(result_dir)
      # Launch the experiment and write the standard output/error
      def is_multi_param(param):
        return any(isinstance(param, typ) for typ in (list, tuple))
      def param_to_str(param):
        if is_multi_param(param):
          return (" ").join(shlex.quote(str(val)) for val in param)
        return shlex.quote(str(param))
      tools.trace("python3 -OO attack.py %s" % (" ").join("--%s %s" % (key, param_to_str(val)) for key, val in params.items()))
      command = ["python3", "-OO", "attack.py"]
      for key, val in params.items():
        command.append("--%s" % (key,))
        if is_multi_param(val):
          for subval in val:
            command.append(str(subval))
        else:
          command.append(str(val))
      cmd_res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      if cmd_res.returncode == 0:
        tools.info("Experiment successful")
      else:
        tools.warning("Experiment failed")
      (result_dir / "stdout.log").write_bytes(cmd_res.stdout)
      (result_dir / "stderr.log").write_bytes(cmd_res.stderr)

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
            name, seed, params = self._jobs.pop()
            break
          # Wait for new job notification
          self._cvready.wait()
      # Run the picked experiment
      self._run(name, seed, device, params)

  def __init__(self, devices=["cpu"], devmult=1, seeds=tuple(range(1, 6))):
    """ Initialize the instance, launch the worker pool.
    Args:
      devices List/tuple of the devices to use in parallel
      devmult How many experiments are run in parallel per device
      seeds   List/tuple of seeds to repeat the experiments with
    """
    # Initialize instance
    self._jobs    = list() # List of tuples (name, seed, params), or None to signal termination
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

  def submit(self, name, params):
    """ Submit an experiment to be run with each seed on any available device.
    Args:
      name   Experiment unique name
      params Named parameters
    """
    with self._lock:
      # Check if not closed
      if self._jobs is None:
        raise RuntimeError("Experiment manager cannot take new jobs as it has been closed")
      # Submit the experiment with each seed
      for seed in self._seeds:
        self._jobs.insert(0, (name, seed, params.copy()))
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

# Jobs
jobs  = Jobs(devices=args.devices, devmult=args.supercharge)
seeds = jobs.get_seeds()

# Base parameters for the MNIST experiments
params_mnist = {
  "dataset": "mnist",
  "batch-size": 83,
  "model": "simples-full",
  "loss": "nll",
  "learning-rate-decay-delta": 300,
  "momentum": 0.9,
  "l2-regularize": 1e-4,
  "evaluation-delta": 5,
  "gradient-clip": 2,
  "nb-steps": 300,
  "nb-for-study": 1,
  "nb-for-study-past": 150,
  "nb-workers": 51
}

# Submit all MNIST experiments
for f, fm in ((24, 2), (12, 3)):
  for lr in (0.5, 0.02):
    # No attack
    params = params_mnist.copy()
    params["nb-workers"] -= f
    params["learning-rate"] = lr
    jobs.submit("mnist-average-n_%s-lr_%s" % (params["nb-workers"], lr), params)
    # Attacked
    for gar in ("krum", "median", "bulyan")[:fm]:
      for attack, attargs in (("little", ("factor:1.5", "negative:True")), ("empire", "factor:1.1")):
        for momentum in ("update", "worker"):
          params = params_mnist.copy()
          params["learning-rate"] = lr
          params["nb-decl-byz"] = params["nb-real-byz"] = f
          params["gar"] = gar
          params["attack"] = attack
          params["attack-args"] = attargs
          params["momentum-at"] = momentum
          jobs.submit("mnist-%s-%s-f_%s-lr_%s-at_%s" % (attack, gar, f, lr, momentum), params)

# Base parameters for the CIFAR-10 experiments
params_cifar10 = {
  "dataset": "cifar10",
  "batch-size": 50,
  "model": "empire-cnn",
  "loss": "nll",
  "learning-rate-decay": 167,
  "momentum": 0.99,
  "l2-regularize": 1e-2,
  "evaluation-delta": 100,
  "gradient-clip": 5,
  "nb-steps": 3000,
  "nb-for-study": 1,
  "nb-for-study-past": 25,
  "nb-workers": 25
}

# Submit all CIFAR-10 experiments
for f, fm in ((11, 2), (5, 3)):
  for lr, dd in ((0.01, 1500), (0.001, 3000)):
    # No attack
    params = params_cifar10.copy()
    params["nb-workers"] -= f
    params["learning-rate"] = lr
    params["learning-rate-decay-delta"] = dd
    jobs.submit("cifar10-average-n_%s-lr_%s" % (params["nb-workers"], lr), params)
    # Attacked
    for gar in ("krum", "median", "bulyan")[:fm]:
      for attack, attargs in (("little", ("factor:1.5", "negative:True")), ("empire", "factor:1.1")):
        for momentum in ("update", "worker"):
          params = params_cifar10.copy()
          params["learning-rate"] = lr
          params["learning-rate-decay-delta"] = dd
          params["nb-decl-byz"] = params["nb-real-byz"] = f
          params["gar"] = gar
          params["attack"] = attack
          params["attack-args"] = attargs
          params["momentum-at"] = momentum
          jobs.submit("cifar10-%s-%s-f_%s-lr_%s-at_%s" % (attack, gar, f, lr, momentum), params)

# Wait for the jobs to finish and close the pool
jobs.wait(exit_is_requested)
jobs.close()

# Check if exit requested before going to plotting the results
if exit_is_requested():
  exit(0)

# ---------------------------------------------------------------------------- #
# Analyze and plot results
tools.success("Analyzing results...")

# Import additional modules
try:
  import study
except ImportError as err:
  tools.fatal("Unable to analyze/plot results: %s" % (err,))

with tools.Context("analysis", "info"):
  # Generate experiment names
  paths = sorted(pathlib.Path("results-data").iterdir())
  # Count number of times ratio condition was validated, for each experiment
  expwith = 0
  expzero = 0
  expmxrt = None
  for path in paths:
    sess = study.Session(path)
    if not sess.has_known_ratio():
      continue
    expwith += 1
    data = sess.compute_ratio().data
    minloss = data["Average loss"][0]
    nbtotal = len(data) - 1  # Does not count last row (which only contains the model top-1 cross-accuracy)
    nbvalid = ((data["Average loss"] <= minloss) & (data["Ratio enough for GAR?"])).sum()  # Exclude the few cases where the model is already "killed"
    nbratio = nbvalid / nbtotal * 100.
    if nbvalid == 0:
      expzero += 1
    else:
      if expmxrt is None or nbratio > expmxrt[2]:
        expmxrt = (nbvalid, nbtotal, nbratio)
    print("· %47s: %4d/%4d (%.2f%%)" % (path.name, nbvalid, nbtotal, nbratio))
  # Print global stats
  print("#experiments with ratio never validated: %4d/%4d (%.2f%%)" % (expzero, expwith, expzero / expwith * 100.))
  print("Maximum #steps with ratio validated:     %4d/%4d (%.2f%%)" % expmxrt)

# ---------------------------------------------------------------------------- #
# Plot results
tools.success("Plotting results...")

# Import additional modules
try:
  import numpy
  import pandas
except ImportError as err:
  tools.fatal("Unable to plot results: %s" % (err,))

def compute_avg_err(name, *cols, avgs="", errs="-err"):
  """ Compute the average and standard deviation of the selected columns over the given experiment.
  Args:
    name Given experiment name
    ...  Selected column names (through 'study.select')
    avgs Suffix for average column names
    errs Suffix for standard deviation (or "error") column names
  Returns:
    Data frames, each for the computed columns
  """
  # Load all the runs for the given experiment name, and keep only a subset
  datas = tuple(study.select(study.Session(args.data_directory / ("%s-%d" % (name, seed))).compute_ratio(nowarn=True), *cols) for seed in seeds)
  # Make the aggregated data frames
  def make_df(col):
    nonlocal datas
    # For every selected columns
    subds = tuple(study.select(data, col).dropna() for data in datas)
    res   = pandas.DataFrame(index=subds[0].index)
    for col in subds[0]:
      # Generate compound column names
      avgn = col + avgs
      errn = col + errs
      # Compute compound columns
      numds = numpy.stack(tuple(subd[col].to_numpy() for subd in subds))
      res[avgn] = numds.mean(axis=0)
      res[errn] = numds.std(axis=0)
    # Return the built data frame
    return res
  # Return the built data frames
  return tuple(make_df(col) for col in cols)

def select_ymax(data_u, data_w):
  """ Select the max y value for the given ratio data.
  Args:
    data_u Ratio for momentum at update
    data_w Ratio for momentum at the workers
  Returns:
    Maximum y value to use in the plot
  """
  vmax = max(data_u[1]["Honest ratio"].max(), data_w[1]["Honest ratio"].max())
  for ymax in (1., 2., 6., 12.):
    if vmax < ymax:
      return ymax
  return 20.

# Plot MNIST results
with tools.Context("mnist", "info"):
  for f, fm in ((24, 2), (12, 3)):
    for lr in (0.5, 0.02):
      # No attack
      name = "mnist-%%s-n_%s-lr_%s" % (params_mnist["nb-workers"] - f, lr)
      gar  = "average"
      try:
        noattack = compute_avg_err(name % gar, "acc", "Honest ratio")
      except Exception as err:
        tools.warning("Unable to process %r: %s" % (name % gar, err))
        continue
      # Attacked
      for attack, attargs in (("little", "factor:1.5 negative:True"), ("empire", "factor:1.1")):
        attacked_at = dict()
        for momentum in ("update", "worker"):
          name = "mnist-%s-%%s-f_%s-lr_%s-at_%s" % (attack, f, lr, momentum)
          attacked = dict()
          for gar in ("krum", "median", "bulyan")[:fm]:
            try:
              attacked[gar] = compute_avg_err(name % gar, "acc", "Honest ratio")
            except Exception as err:
              tools.warning("Unable to process %r: %s" % (name % gar, err))
              continue
          attacked_at[momentum] = attacked
          # Plot top-1 cross-accuracy
          plot = study.LinePlot()
          plot.include(noattack[0], "acc", errs="-err", lalp=0.8)
          legend = ["No attack"]
          for gar in ("krum", "median", "bulyan")[:fm]:
            if gar not in attacked:
              continue
            plot.include(attacked[gar][0], "acc", errs="-err", lalp=0.8)
            legend.append(gar.capitalize())
          plot.finalize(None, "Step number", "Top-1 cross-accuracy", xmin=0, xmax=300, ymin=0, ymax=1, legend=legend)
          plot.save(args.plot_directory / ("mnist-%s-f_%s-lr_%s-at_%s.png" % (attack, f, lr, momentum)), xsize=3, ysize=1.5)
        # Plot per-gar variance-norm ratios
        for gar in ("krum", "median", "bulyan")[:fm]:
          data_u = attacked_at["update"].get(gar)
          data_w = attacked_at["worker"].get(gar)
          if data_u is None or data_w is None:
            continue
          plot = study.LinePlot()
          plot.include(data_u[1], "ratio", errs="-err", lalp=0.5, ccnt=0)
          plot.include(data_w[1], "ratio", errs="-err", lalp=0.5, ccnt=4)
          plot.finalize(None, "Step number", "Variance-norm ratio", xmin=0, xmax=300, ymin=0, ymax=select_ymax(data_u, data_w), legend=tuple("%s @%s" % (gar.capitalize(), at) for at in ("server", "worker")))
          plot.save(args.plot_directory / ("mnist-%s-%s-f_%s-lr_%s-ratio.png" % (attack, gar, f, lr)), xsize=3, ysize=1.5)

# Plot CIFAR-10 results
with tools.Context("cifar10", "info"):
  for f, fm in ((11, 2), (5, 3)):
    for lr in (0.01, 0.001):
      # No attack
      name = "cifar10-%%s-n_%s-lr_%s" % (params_cifar10["nb-workers"] - f, lr)
      gar  = "average"
      try:
        noattack = compute_avg_err(name % gar, "acc", "Honest ratio")
      except Exception as err:
        tools.warning("Unable to process %r: %s" % (name % gar, err))
        continue
      # Attacked
      for attack, attargs in (("little", "factor:1.5 negative:True"), ("empire", "factor:1.1")):
        attacked_at = dict()
        for momentum in ("update", "worker"):
          name = "cifar10-%s-%%s-f_%s-lr_%s-at_%s" % (attack, f, lr, momentum)
          attacked = dict()
          for gar in ("krum", "median", "bulyan")[:fm]:
            try:
              attacked[gar] = compute_avg_err(name % gar, "acc", "Honest ratio")
            except Exception as err:
              tools.warning("Unable to process %r: %s" % (name % gar, err))
              continue
          attacked_at[momentum] = attacked
          # Plot top-1 cross-accuracy
          plot = study.LinePlot()
          plot.include(noattack[0], "acc", errs="-err", lalp=0.8)
          legend = ["No attack"]
          for gar in ("krum", "median", "bulyan")[:fm]:
            if gar not in attacked:
              continue
            plot.include(attacked[gar][0], "acc", errs="-err", lalp=0.8)
            legend.append(gar.capitalize())
          plot.finalize(None, "Step number", "Top-1 cross-accuracy", xmin=0, xmax=3000, ymin=0, ymax=0.9, legend=legend)
          plot.save(args.plot_directory / ("cifar10-%s-f_%s-lr_%s-at_%s.png" % (attack, f, lr, momentum)), xsize=3, ysize=1.5)
        # Plot per-gar variance-norm ratios
        for gar in ("krum", "median", "bulyan")[:fm]:
          data_u = attacked_at["update"].get(gar)
          data_w = attacked_at["worker"].get(gar)
          if data_u is None or data_w is None:
            continue
          plot = study.LinePlot()
          plot.include(data_u[1], "ratio", errs="-err", lalp=0.5, ccnt=0)
          plot.include(data_w[1], "ratio", errs="-err", lalp=0.5, ccnt=4)
          plot.finalize(None, "Step number", "Variance-norm ratio", xmin=0, xmax=3000, ymin=0, ymax=select_ymax(data_u, data_w), legend=tuple("%s @%s" % (gar.capitalize(), at) for at in ("server", "worker")))
          plot.save(args.plot_directory / ("cifar10-%s-%s-f_%s-lr_%s-ratio.png" % (attack, gar, f, lr)), xsize=3, ysize=1.5)

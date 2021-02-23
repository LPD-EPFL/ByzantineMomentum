# coding: utf-8
###
 # @file   reproduce-appendix.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Reproduce the (missing) experiments and plots (supplementary experiments).
###

import tools
tools.success("Module loading...")

import argparse
import pathlib
import signal
import sys

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
    default="results-data-appendix",
    help="Path of the data directory, containing the data gathered from the experiments")
  parser.add_argument("--plot-directory",
    type=str,
    default="results-plot-appendix",
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
    tools.fatal(f"Expected a positive supercharge value, got {args.supercharge}")
  # Make the result directories
  def check_make_dir(path):
    path = pathlib.Path(path)
    if path.exists():
      if not path.is_dir():
        tools.fatal(f"Given path {str(path)!r} must point to a directory")
    else:
      path.mkdir(mode=0o755, parents=True)
    return path
  args.data_directory = check_make_dir(args.data_directory)
  args.plot_directory = check_make_dir(args.plot_directory)
  # Preprocess/resolve the devices to use
  if args.devices == "auto":
    if torch.cuda.is_available():
      args.devices = list(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    else:
      args.devices = ["cpu"]
  else:
    args.devices = list(name.strip() for name in args.devices.split(","))

# ---------------------------------------------------------------------------- #
# Serial preloading of the dataset
tools.success("Pre-downloading datasets...")

# Pre-load the datasets to prevent the first parallel runs from downloading them several times
with tools.Context("dataset", "info"):
  for name in ("cifar10",):
    with tools.Context(name, "info"):
      experiments.make_datasets(name, 1, 1)

# ---------------------------------------------------------------------------- #
# Run (missing) experiments
tools.success("Running experiments...")

# GAR to use
gars = ("krum", "median", "bulyan")

# Command maker helper
def make_command(params):
  cmd = ["python3", "-OO", "attack.py"]
  cmd += tools.dict_to_cmdlist(params)
  return tools.Command(cmd)

# Jobs
jobs  = tools.Jobs(args.data_directory, devices=args.devices, devmult=args.supercharge)
seeds = jobs.get_seeds()

# Base parameters for the CIFAR-10 experiments
params_cifar10 = {
  "batch-size": 20,
  "model": "wide_resnet-Wide_ResNet",
  "model-args": ("depth:28", "widen_factor:10", "dropout_rate:0.3", "num_classes:10"),
  "learning-rate-schedule": "0.02, 8000, 0.004, 16000, 0.0008",
  "gradient-clip": 5,
  "loss": "crossentropy",
  "momentum": 0.99,
  "momentum-nesterov": True,
  "l2-regularize": 5e-4,
  "evaluation-delta": 100,
  "nb-steps": 20000,
  "nb-for-study": 1,
  "nb-for-study-past": 1,
  "nb-workers": 11
}

# Submit all CIFAR-10 experiments
for ds in ("cifar10",):
  for f, fm in ((4, 1), (2, 0)):
    # No attack
    params = params_cifar10.copy()
    params["dataset"] = ds
    params["nb-workers"] -= f
    jobs.submit(f"{ds}-average-n_{params['nb-workers']}-lr_pow-nesterov", make_command(params))
    # Attacked
    for gar in gars[:len(gars) - fm]:
      for attack, attargs in (("little", ("factor:1.5", "negative:True")), ("empire", "factor:1.1")):
        for momentum in ("update", "worker"):
          params = params_cifar10.copy()
          params["dataset"] = ds
          params["nb-decl-byz"] = params["nb-real-byz"] = f
          params["gar"] = gar
          params["attack"] = attack
          params["attack-args"] = attargs
          params["momentum-at"] = momentum
          jobs.submit(f"{ds}-{attack}-{gar}-f_{f}-lr_pow-at_{momentum}-nesterov", make_command(params))

# Wait for the jobs to finish and close the pool
jobs.wait(exit_is_requested)
jobs.close()

# Check if exit requested before going to plotting the results
if exit_is_requested():
  exit(0)

# ---------------------------------------------------------------------------- #
# Plot results
tools.success("Plotting results...")

# Import additional modules
try:
  import numpy
  import pandas
  import study
except ImportError as err:
  tools.fatal(f"Unable to plot results: {err}")

# Name mapping in overview plots
overview_names = {
  "update": "Standard\nformulation",
  "worker": "Our\nformulation" }

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
  datas = list()
  for seed in seeds:
    try:
      sess = study.Session(args.data_directory / f"{name}-{seed}")
    except:
      continue
    datas.append(study.select(sess.compute_ratio(nowarn=True), *cols))
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

def get_max_accuracies(name):
  """ Map each seed to the maximum accuracy reached by the given experiment name.
  Args:
    name Given experiment name
  Returns:
    Generator of maximum accuracies
  """
  for seed in seeds:
    try:
      sess = study.Session(args.data_directory / f"{name}-{seed}")
    except:
      continue
    yield sess.get("accuracy").max().values.item()

def median(iterable):
  """ Get the median of the values in the given once-iterable.
  Args:
    iterable Generator-like iterable to recover the median from
  Returns:
    Median value, None if the generator yielded no value
  """
  # Consume the values
  data = list(iterable)
  l = len(data)
  # Fast path if no value
  if l == 0:
    return None
  # Sort the data
  data.sort()
  # Recover the median
  m = l // 2
  if l % 2 == 0:
    return (data[m - 1] + data[m]) / 2
  else:
    return data[m]

def select_ymax(data_w):
  """ Select the max y value for the given ratio data.
  Args:
    data_w Ratio data
  Returns:
    Maximum y value to use in the plot
  """
  vmax = max(data_w[3]["Sampled ratio"].max(), data_w[1]["Honest ratio"].max())
  for ymax in (1., 2., 6., 12.):
    if vmax < ymax:
      return ymax
  return 20.

# Plot CIFAR-10/100 results
for ds in ("cifar10",):
  with tools.Context(ds, "info"):
    for f, fm in ((4, 1), (2, 0)):
      # No attack
      name = f"{ds}-%s-n_{params_cifar10['nb-workers'] - f}-lr_pow-nesterov"
      gar  = "average"
      try:
        noattack = compute_avg_err(name % gar, "Accuracy", "Honest ratio", "Average loss")
      except Exception as err:
        tools.warning(f"Unable to process {name % gar!r}: {err}")
        continue
      # Attacked
      for attack, attargs in (("little", "factor:1.5 negative:True"), ("empire", "factor:1.1")):
        attacked_at = dict()
        for momentum in ("update", "worker"):
          name = f"{ds}-{attack}-%s-f_{f}-lr_pow-at_{momentum}-nesterov"
          attacked = dict()
          for gar in gars[:len(gars) - fm]:
            try:
              attacked[gar] = compute_avg_err(name % gar, "Accuracy", "Honest ratio", "Average loss", "Sampled ratio")
            except Exception as err:
              tools.warning(f"Unable to process {name % gar!r}: {err}")
              continue
          attacked_at[momentum] = attacked
          # Plot top-1 cross-accuracy
          plot = study.LinePlot()
          plot.include(noattack[0], "Accuracy", errs="-err", lalp=0.8)
          legend = ["No attack"]
          for gar in gars[:len(gars) - fm]:
            if gar not in attacked:
              continue
            plot.include(attacked[gar][0], "Accuracy", errs="-err", lalp=0.8)
            legend.append(gar.capitalize())
          plot.finalize(None, "Step number", "Top-1 cross-accuracy", xmin=0, xmax=params_cifar10["nb-steps"], ymin=0, ymax=0.9, legend=legend)
          plot.save(args.plot_directory / f"{ds}-{attack}-f_{f}-lr_pow-at_{momentum}-nesterov.png", xsize=3, ysize=1.5)
          # Plot average loss
          plot = study.LinePlot()
          plot.include(noattack[2], "Average loss", errs="-err", lalp=0.8)
          legend = ["No attack"]
          for gar in gars[:len(gars) - fm]:
            if gar not in attacked:
              continue
            plot.include(attacked[gar][2], "Average loss", errs="-err", lalp=0.8)
            legend.append(gar.capitalize())
          plot.finalize(None, "Step number", "Average loss", xmin=0, xmax=params_cifar10["nb-steps"], ymin=0, legend=legend)
          plot.save(args.plot_directory / f"{ds}-{attack}-f_{f}-lr_pow-at_{momentum}-nesterov-loss.png", xsize=3, ysize=1.5)
        # Plot per-gar variance-norm ratios
        for gar in gars[:len(gars) - fm]:
          data_w = attacked_at["worker"].get(gar)
          if data_w is None:
            continue
          plot = study.LinePlot()
          plot.include(data_w[3], "ratio", errs="-err", lalp=0.5, ccnt=0)
          plot.include(data_w[1], "ratio", errs="-err", lalp=0.5, ccnt=4)
          plot.finalize(None, "Step number", "Variance-norm ratio", xmin=0, xmax=params_cifar10["nb-steps"], ymin=0, ymax=select_ymax(data_w), legend=tuple(f"{gar.capitalize()} \"{at}\"" for at in ("sample", "submit")))
          plot.save(args.plot_directory / f"{ds}-{attack}-{gar}-f_{f}-lr_pow-nesterov-ratio.png", xsize=3, ysize=1.5)

# Plot CIFAR-10/100 results
for ds in ("cifar10",):
  with tools.Context(ds, "info"):
    for f, fm in ((4, 1), (2, 0)):
       # Get median of unattacked max top-1 cross-accuracy
      ref = median(get_max_accuracies(f"{ds}-average-n_{params_cifar10['nb-workers'] - f}-lr_pow-nesterov"))
      # Attacked
      for attack, _ in (("little", "factor:1.5 negative:True"), ("empire", "factor:1.1")):
        attacked_at = dict()
        for momentum in ("update", "worker"):
          name = f"{ds}-{attack}-%s-f_{f}-lr_pow-at_{momentum}-nesterov"
          attacked = list()
          for gar in gars[:len(gars) - fm]:
            try:
              attacked.extend(get_max_accuracies(name % gar))
            except Exception as err:
              tools.warning(f"Unable to process {name % gar!r}: {err}")
              continue
          attacked_at[momentum] = numpy.array(attacked)
        # Plot maximum top-1 cross-accuracies
        plot = study.BoxPlot()
        for momentum in ("update", "worker"):
          plot.include(attacked_at[momentum], overview_names.get(momentum, f"At {momentum}"))
        plot.hline(ref)
        plot.finalize(None, "Max. top-1 cross-accuracy", ymin=0, ymax=1)
        plot.save(args.plot_directory / f"overview-{ds}-{attack}-f_{f}-lr_pow-nesterov.png", xsize=1.5, ysize=1.5)

# coding: utf-8
###
 # @file   reproduce.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Reproduce the (missing) experiments and plots.
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
  for name in ("mnist", "fashionmnist", "cifar10", "cifar100"):
    with tools.Context(name, "info"):
      experiments.make_datasets(name, 1, 1)

# ---------------------------------------------------------------------------- #
# Run (missing) experiments
tools.success("Running experiments...")

# GAR to use
gars = ("krum", "median", "trmean", "phocas", "meamed", "bulyan")

# Command maker helper
def make_command(params):
  cmd = ["python3", "-OO", "attack.py"]
  cmd += tools.dict_to_cmdlist(params)
  return tools.Command(cmd)

# Jobs
jobs  = tools.Jobs(args.data_directory, devices=args.devices, devmult=args.supercharge)
seeds = jobs.get_seeds()

# Base parameters for the (Fashion-)MNIST experiments
params_mnist = {
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

# Submit all (Fashion-)MNIST experiments
for ds in ("mnist", "fashionmnist"):
  for f, fm in ((24, 1), (12, 0)):
    for lr in (0.5, 0.02):
      for nesterov in (False, True):
        # No attack
        params = params_mnist.copy()
        params["dataset"] = ds
        params["nb-workers"] -= f
        params["learning-rate"] = lr
        params["momentum-nesterov"] = nesterov
        jobs.submit(f"{ds}-average-n_{params['nb-workers']}-lr_{lr}" + ("-nesterov" if nesterov else ""), make_command(params))
        # Attacked
        for gar in gars[:len(gars) - fm]:
          for attack, attargs in (("little", ("factor:1.5", "negative:True")), ("empire", "factor:1.1")):
            for momentum in ("update", "worker"):
              params = params_mnist.copy()
              params["dataset"] = ds
              params["learning-rate"] = lr
              params["nb-decl-byz"] = params["nb-real-byz"] = f
              params["gar"] = gar
              params["attack"] = attack
              params["attack-args"] = attargs
              params["momentum-at"] = momentum
              params["momentum-nesterov"] = nesterov
              jobs.submit(f"{ds}-{attack}-{gar}-f_{f}-lr_{lr}-at_{momentum}" + ("-nesterov" if nesterov else ""), make_command(params))

# Base parameters for the CIFAR-10 experiments
params_cifar10 = {
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

# Submit all CIFAR-10/100 experiments
for ds, mp in (("cifar10", "cifar100:False"), ("cifar100", "cifar100:True")):
  for f, fm in ((11, 1), (5, 0)):
    for lr, dd in ((0.01, 1500), (0.001, 3000)):
      for nesterov in (False, True):
        # No attack
        params = params_cifar10.copy()
        params["dataset"] = ds
        params["model-args"] = mp
        params["nb-workers"] -= f
        params["learning-rate"] = lr
        params["learning-rate-decay-delta"] = dd
        params["momentum-nesterov"] = nesterov
        jobs.submit(f"{ds}-average-n_{params['nb-workers']}-lr_{lr}" + ("-nesterov" if nesterov else ""), make_command(params))
        # Attacked
        for gar in gars[:len(gars) - fm]:
          for attack, attargs in (("little", ("factor:1.5", "negative:True")), ("empire", "factor:1.1")):
            for momentum in ("update", "worker"):
              params = params_cifar10.copy()
              params["dataset"] = ds
              params["model-args"] = mp
              params["learning-rate"] = lr
              params["learning-rate-decay-delta"] = dd
              params["nb-decl-byz"] = params["nb-real-byz"] = f
              params["gar"] = gar
              params["attack"] = attack
              params["attack-args"] = attargs
              params["momentum-at"] = momentum
              params["momentum-nesterov"] = nesterov
              jobs.submit(f"{ds}-{attack}-{gar}-f_{f}-lr_{lr}-at_{momentum}" + ("-nesterov" if nesterov else ""), make_command(params))

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
  tools.fatal(f"Unable to analyze/plot results: {err}")

def get_reference_accuracy(name):
  """ Get the maximum accuracy obtained by the associated, unattacked run.
  Args:
    name Name of the attacked run
  Returns:
    Maximum accuracy of the associated, unattacked run; None if name not an attack run
  """
  global maxaccs
  # Analyze name
  elems = name.split("-")
  if len(elems) < 6:
    return None
  if elems[3][:2] != "f_":
    return None
  nbbyz = int(elems[3][2:])
  # Build associated name
  if elems[-2] == "nesterov":
    queue = f"nesterov-{elems[-1]}"
  else:
    queue = elems[-1]
  aname = f"{elems[0]}-average-n_{get_reference_accuracy.nbwrks[elems[0]] - nbbyz}-{elems[4]}-{queue}"
  # Return associated accuracy
  return maxaccs[aname]
get_reference_accuracy.nbwrks = {
  "mnist": params_mnist["nb-workers"],
  "fashionmnist": params_mnist["nb-workers"],
  "cifar10": params_cifar10["nb-workers"],
  "cifar100": params_cifar10["nb-workers"] }

with tools.Context("analysis", "info"):
  # Generate experiment names
  def is_complete(path):
    return all(content not in str(path) for content in (".failed", ".pending"))
  paths = sorted(filter(is_complete, pathlib.Path("results-data").iterdir()))
  maxlen = max(len(path.name) for path in paths)
  maxaccs = dict()
  # Count number of times ratio condition was validated, for each experiment
  expwith = 0
  expzero = 0
  expmxrt = None
  for path in paths:
    sess = study.Session(path)
    maxaccs[path.name] = sess.get("accuracy").max().values.item()
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
    print(f"· {path.name:{maxlen}s}: {nbvalid:4d}/{nbtotal:4d} ({nbratio:.2f}%)")
  # Print global stats
  print(f"#experiments with ratio never validated: {expzero:4d}/{expwith:4d} ({expzero / expwith * 100.:.2f}%)")
  if expmxrt is None:
    expmxrt = "<no data>"
  else:
    expmxrt = f"{nbvalid:4d}/{nbtotal:4d} ({nbratio:.2f}%)"
  print(f"Maximum #steps with ratio validated:     {expmxrt}")
  # Compute general max accuracy improvement for several subsets
  for subset in (None, "mnist", "cifar", "fashion", "f_24", "f_12", "cifar10-", "cifar100", "f_11", "f_5"):
    with tools.Context("everything" if subset is None else subset, None):
      mactotal    = 0
      maceffect10 = 0
      maceffect20 = 0
      maceffect40 = 0
      macabove10  = 0
      macabove20  = 0
      macabove40  = 0
      macbad0     = 0
      macbad02    = 0
      macbad05    = 0
      macloss05   = 0
      macloss10   = 0
      for path in paths:
        name = path.name
        # Select only "at-worker" version
        if "at_worker" not in name:
          continue
        # Select only the current subset
        if subset is not None and subset not in name:
          continue
        # Recover max accuracies
        ref = get_reference_accuracy(name)
        ats = maxaccs[name.replace("at_worker", "at_update")]
        atw = maxaccs[name]
        # Analyze
        mactotal += 1
        loss = ref - ats
        gain = atw - ats
        if gain < 0:
          macbad0 += 1
          if gain < -0.02:
            macbad02 += 1
          if gain < -0.05:
            macbad05 += 1
          if ref - atw > 0.05:
            macloss05 += 1
          if ref - atw > 0.1:
            macloss10 += 1
        if loss > 0.1:
          maceffect10 += 1
          if gain > 0.1:
            macabove10 += 1
        if loss > 0.2:
          maceffect20 += 1
          if gain > 0.2:
            macabove20 += 1
        if loss > 0.4:
          maceffect40 += 1
          if gain > 0.4:
            macabove40 += 1
      # Print improvement stats
      print(f"#experiments with effective attack (10%): {maceffect10:4d}/{mactotal:4d} ({maceffect10 / mactotal * 100.:.2f}%)")
      print(f"#experiments with effective attack (20%): {maceffect20:4d}/{mactotal:4d} ({maceffect20 / mactotal * 100.:.2f}%)")
      print(f"#experiments with effective attack (40%): {maceffect40:4d}/{mactotal:4d} ({maceffect40 / mactotal * 100.:.2f}%)")
      if maceffect10 > 0:
        print(f"#experiments with defense gain above 10%: {macabove10:4d}/{maceffect10:4d} ({macabove10 / maceffect10 * 100.:.2f}%)")
      else:
        print(f"#experiments with defense gain above 10%:    N/A")
      if maceffect20 > 0:
        print(f"#experiments with defense gain above 20%: {macabove20:4d}/{maceffect20:4d} ({macabove20 / maceffect20 * 100.:.2f}%)")
      else:
        print(f"#experiments with defense gain above 20%:    N/A")
      if maceffect40 > 0:
        print(f"#experiments with defense gain above 40%: {macabove40:4d}/{maceffect40:4d} ({macabove40 / maceffect40 * 100.:.2f}%)")
      else:
        print(f"#experiments with defense gain above 40%:    N/A")
      print(f"#experiments with >0% performance loss:   {macbad0:4d}/{mactotal:4d} ({macbad0 / mactotal * 100.:.2f}%)")
      print(f"#experiments with >2% performance loss:   {macbad02:4d}/{mactotal:4d} ({macbad02 / mactotal * 100.:.2f}%)")
      print(f"#experiments with >5% performance loss:   {macbad05:4d}/{mactotal:4d} ({macbad05 / mactotal * 100.:.2f}%)")
      print(f"#experiments with >5% \"optimality\" loss:  {macloss05:4d}/{mactotal:4d} ({macloss05 / mactotal * 100.:.2f}%)")
      print(f"#experiments with >10% \"optimality\" loss: {macloss10:4d}/{mactotal:4d} ({macloss10 / mactotal * 100.:.2f}%)")

# ---------------------------------------------------------------------------- #
# Plot results
tools.success("Plotting results...")

# Import additional modules
try:
  import numpy
  import pandas
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
  datas = tuple(study.select(study.Session(args.data_directory / f"{name}-{seed}").compute_ratio(nowarn=True), *cols) for seed in seeds)
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
  return (study.select(study.Session(args.data_directory / f"{name}-{seed}").get("accuracy").max().values.item()) for seed in seeds)

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

# Plot (Fashion-)MNIST results
for ds in ("mnist", "fashionmnist"):
  with tools.Context(ds, "info"):
    for nesterov in (False, True):
      for f, fm in ((24, 1), (12, 0)):
        for lr in (0.5, 0.02):
          # No attack
          name = f"{ds}-%s-n_{params_mnist['nb-workers'] - f}-lr_{lr}" + ("-nesterov" if nesterov else "")
          gar  = "average"
          try:
            noattack = compute_avg_err(name % gar, "Accuracy", "Honest ratio", "Average loss")
          except Exception as err:
            tools.warning(f"Unable to process {name % gar}: {err}")
            continue
          # Attacked
          for attack, attargs in (("little", "factor:1.5 negative:True"), ("empire", "factor:1.1")):
            attacked_at = dict()
            for momentum in ("update", "worker"):
              name = f"{ds}-{attack}-%s-f_{f}-lr_{lr}-at_{momentum}" + ("-nesterov" if nesterov else "")
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
              plot.finalize(None, "Step number", "Top-1 cross-accuracy", xmin=0, xmax=300, ymin=0, ymax=1, legend=legend)
              plot.save(args.plot_directory / f"{ds}-{attack}-f_{f}-lr_{lr}-at_{momentum}{'-nesterov' if nesterov else ''}.png", xsize=3, ysize=1.5)
              # Plot average loss
              plot = study.LinePlot()
              plot.include(noattack[2], "Average loss", errs="-err", lalp=0.8)
              legend = ["No attack"]
              for gar in gars[:len(gars) - fm]:
                if gar not in attacked:
                  continue
                plot.include(attacked[gar][2], "Average loss", errs="-err", lalp=0.8)
                legend.append(gar.capitalize())
              plot.finalize(None, "Step number", "Average loss", xmin=0, xmax=300, ymin=0, legend=legend)
              plot.save(args.plot_directory / f"{ds}-{attack}-f_{f}-lr_{lr}-at_{momentum}{'-nesterov' if nesterov else ''}-loss.png", xsize=3, ysize=1.5)
            # Plot per-gar variance-norm ratios
            for gar in gars[:len(gars) - fm]:
              data_w = attacked_at["worker"].get(gar)
              if data_w is None:
                continue
              plot = study.LinePlot()
              plot.include(data_w[3], "ratio", errs="-err", lalp=0.5, ccnt=0)
              plot.include(data_w[1], "ratio", errs="-err", lalp=0.5, ccnt=4)
              plot.finalize(None, "Step number", "Variance-norm ratio", xmin=0, xmax=300, ymin=0, ymax=select_ymax(data_w), legend=tuple(f"{gar.capitalize()} \"{at}\"" for at in ("sample", "submit")))
              plot.save(args.plot_directory / f"{ds}-{attack}-{gar}-f_{f}-lr_{lr}{'-nesterov' if nesterov else ''}-ratio.png", xsize=3, ysize=1.5)

# Plot CIFAR-10/100 results
for ds in ("cifar10", "cifar100"):
  with tools.Context(ds, "info"):
    for nesterov in (False, True):
      for f, fm in ((11, 1), (5, 0)):
        for lr in (0.01, 0.001):
          # No attack
          name = f"{ds}-%s-n_{params_cifar10['nb-workers'] - f}-lr_{lr}" + ("-nesterov" if nesterov else "")
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
              name = f"{ds}-{attack}-%s-f_{f}-lr_{lr}-at_{momentum}" + ("-nesterov" if nesterov else "")
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
              plot.finalize(None, "Step number", "Top-1 cross-accuracy", xmin=0, xmax=3000, ymin=0, ymax=0.9, legend=legend)
              plot.save(args.plot_directory / f"{ds}-{attack}-f_{f}-lr_{lr}-at_{momentum}{'-nesterov' if nesterov else ''}.png", xsize=3, ysize=1.5)
              # Plot average loss
              plot = study.LinePlot()
              plot.include(noattack[2], "Average loss", errs="-err", lalp=0.8)
              legend = ["No attack"]
              for gar in gars[:len(gars) - fm]:
                if gar not in attacked:
                  continue
                plot.include(attacked[gar][2], "Average loss", errs="-err", lalp=0.8)
                legend.append(gar.capitalize())
              plot.finalize(None, "Step number", "Average loss", xmin=0, xmax=3000, ymin=0, legend=legend)
              plot.save(args.plot_directory / f"{ds}-{attack}-f_{f}-lr_{lr}-at_{momentum}{'-nesterov' if nesterov else ''}-loss.png", xsize=3, ysize=1.5)
            # Plot per-gar variance-norm ratios
            for gar in gars[:len(gars) - fm]:
              data_w = attacked_at["worker"].get(gar)
              if data_w is None:
                continue
              plot = study.LinePlot()
              plot.include(data_w[3], "ratio", errs="-err", lalp=0.5, ccnt=0)
              plot.include(data_w[1], "ratio", errs="-err", lalp=0.5, ccnt=4)
              plot.finalize(None, "Step number", "Variance-norm ratio", xmin=0, xmax=3000, ymin=0, ymax=select_ymax(data_w), legend=tuple(f"{gar.capitalize()} \"{at}\"" for at in ("sample", "submit")))
              plot.save(args.plot_directory / f"{ds}-{attack}-{gar}-f_{f}-lr_{lr}{'-nesterov' if nesterov else ''}-ratio.png", xsize=3, ysize=1.5)

# Plot (Fashion-)MNIST results
for ds in ("mnist", "fashionmnist"):
  with tools.Context(ds, "info"):
    for nesterov in (False, True):
      for f, fm in ((24, 1), (12, 0)):
        for lr in (0.5, 0.02):
          # Get median of unattacked max top-1 cross-accuracy
          ref = median(get_max_accuracies(f"{ds}-average-n_{params_mnist['nb-workers'] - f}-lr_{lr}" + ("-nesterov" if nesterov else "")))
          # Attacked
          for attack, _ in (("little", "factor:1.5 negative:True"), ("empire", "factor:1.1")):
            attacked_at = dict()
            for momentum in ("update", "worker"):
              name = f"{ds}-{attack}-%s-f_{f}-lr_{lr}-at_{momentum}" + ("-nesterov" if nesterov else "")
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
            plot.save(args.plot_directory / f"overview-{ds}-{attack}-f_{f}-lr_{lr}{'-nesterov' if nesterov else ''}.png", xsize=1.5, ysize=1.5)

# Plot CIFAR-10/100 results
for ds in ("cifar10", "cifar100"):
  with tools.Context(ds, "info"):
    for nesterov in (False, True):
      for f, fm in ((11, 1), (5, 0)):
        for lr in (0.01, 0.001):
           # Get median of unattacked max top-1 cross-accuracy
          ref = median(get_max_accuracies(f"{ds}-average-n_{params_cifar10['nb-workers'] - f}-lr_{lr}" + ("-nesterov" if nesterov else "")))
          # Attacked
          for attack, _ in (("little", "factor:1.5 negative:True"), ("empire", "factor:1.1")):
            attacked_at = dict()
            for momentum in ("update", "worker"):
              name = f"{ds}-{attack}-%s-f_{f}-lr_{lr}-at_{momentum}" + ("-nesterov" if nesterov else "")
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
            plot.save(args.plot_directory / f"overview-{ds}-{attack}-f_{f}-lr_{lr}{'-nesterov' if nesterov else ''}.png", xsize=1.5, ysize=1.5)

# coding: utf-8
###
 # @file   attack.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Simulate a training session under attack.
###

import tools
tools.success("Module loading...")

import argparse
import collections
import json
import math
import os
import pathlib
import random
import signal
import sys
import torch
import torchvision
import traceback

import aggregators
import attacks
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
  parser.add_argument("--seed",
    type=int,
    default=-1,
    help="Fixed seed to use for reproducibility purpose, negative for random seed")
  parser.add_argument("--device",
    type=str,
    default="auto",
    help="Device on which to run the experiment, \"auto\" by default")
  parser.add_argument("--device-gar",
    type=str,
    default="same",
    help="Device on which to run the GAR, \"same\" for no change of device")
  parser.add_argument("--nb-steps",
    type=int,
    default=-1,
    help="Number of (additional) training steps to do, negative for no limit")
  parser.add_argument("--nb-workers",
    type=int,
    default=11,
    help="Total number of worker machines")
  parser.add_argument("--nb-for-study",
    type=int,
    default=11,
    help="Number of gradients to compute for gradient study purpose only, non-positive for no study even when the result directory is set")
  parser.add_argument("--nb-for-study-past",
    type=int,
    default=20,
    help="Number of past gradients to keep for gradient study purpose only, ignored if no study")
  parser.add_argument("--nb-decl-byz",
    type=int,
    default=4,
    help="Number of Byzantine worker(s) to support")
  parser.add_argument("--nb-real-byz",
    type=int,
    default=0,
    help="Number of actual Byzantine worker(s)")
  parser.add_argument("--init-multi",
    type=str,
    default=None,
    help="Model multi-dimensional parameters initialization algorithm; use PyTorch's default if unspecified")
  parser.add_argument("--init-multi-args",
    nargs="*",
    help="Additional initialization algorithm-dependent arguments to pass when initializing multi-dimensional parameters")
  parser.add_argument("--init-mono",
    type=str,
    default=None,
    help="Model mono-dimensional parameters initialization algorithm; use PyTorch's default if unspecified")
  parser.add_argument("--init-mono-args",
    nargs="*",
    help="Additional initialization algorithm-dependent arguments to pass when initializing mono-dimensional parameters")
  parser.add_argument("--gar",
    type=str,
    default="average",
    help="(Byzantine-resilient) aggregation rule to use")
  parser.add_argument("--gar-args",
    nargs="*",
    help="Additional GAR-dependent arguments to pass to the aggregation rule")
  parser.add_argument("--gars",
    type=str,
    default=None,
    help="JSON-string specifying several GARs to use randomly at each step; overrides '--gar' and '--gar-args' if specified")
  parser.add_argument("--attack",
    type=str,
    default="nan",
    help="Attack to use")
  parser.add_argument("--attack-args",
    nargs="*",
    help="Additional attack-dependent arguments to pass to the attack")
  parser.add_argument("--model",
    type=str,
    default="simples-conv",
    help="Model to train")
  parser.add_argument("--model-args",
    nargs="*",
    help="Additional model-dependent arguments to pass to the model")
  parser.add_argument("--loss",
    type=str,
    default="nll",
    help="Loss to use")
  parser.add_argument("--loss-args",
    nargs="*",
    help="Additional loss-dependent arguments to pass to the loss")
  parser.add_argument("--criterion",
    type=str,
    default="top-k",
    help="Criterion to use")
  parser.add_argument("--criterion-args",
    nargs="*",
    help="Additional criterion-dependent arguments to pass to the criterion")
  parser.add_argument("--dataset",
    type=str,
    default="mnist",
    help="Dataset to use")
  parser.add_argument("--batch-size",
    type=int,
    default=25,
    help="Batch-size to use for training")
  parser.add_argument("--batch-size-test",
    type=int,
    default=100,
    help="Batch-size to use for testing")
  parser.add_argument("--batch-size-test-reps",
    type=int,
    default=100,
    help="How many evaluation(s) with the test batch-size to perform")
  parser.add_argument("--no-transform",
    action="store_true",
    default=False,
    help="Whether to disable any dataset tranformation (e.g. random flips)")
  parser.add_argument("--learning-rate",
    type=float,
    default=0.01,
    help="Learning rate to use for training")
  parser.add_argument("--learning-rate-decay",
    type=int,
    default=5000,
    help="Learning rate hyperbolic half-decay time, non-positive for no decay")
  parser.add_argument("--learning-rate-decay-delta",
    type=int,
    default=1,
    help="How many steps between two learning rate updates, must be a positive integer")
  parser.add_argument("--learning-rate-schedule",
    type=str,
    default=None,
    help="Learning rate schedule, format: <init lr>[,<from step>,<new lr>]*; if set, supersede the other '--learning-rate' options")
  parser.add_argument("--momentum",
    type=float,
    default=0.9,
    help="Momentum to use for training")
  parser.add_argument("--dampening",
    type=float,
    default=0.,
    help="Dampening to use for training")
  parser.add_argument("--momentum-nesterov",
    action="store_true",
    default=False,
    help="Use Nesterov's momentum instead of the \"classical\" formulation (see 'torch.optim.SGD')")
  parser.add_argument("--momentum-at",
    type=str,
    default="update",
    help="Where to apply the momentum & dampening ('update': just after the GAR, 'server': just before the GAR, 'worker': at each worker)")
  parser.add_argument("--weight-decay",
    type=float,
    default=0,
    help="Weight decay to use for training")
  parser.add_argument("--l1-regularize",
    type=float,
    default=None,
    help="Add L1 regularization of the given factor to the loss")
  parser.add_argument("--l2-regularize",
    type=float,
    default=None,
    help="Add L2 regularization of the given factor to the loss")
  parser.add_argument("--gradient-clip",
    type=float,
    default=None,
    help="Maximum L2-norm, above which clipping occurs, for the estimated gradients")
  parser.add_argument("--nb-local-steps",
    type=int,
    default=1,
    help="Positive integer, number of local training steps to perform to make a gradient (1 = standard SGD)")
  parser.add_argument("--load-checkpoint",
    type=str,
    default=None,
    help="Load a given checkpoint to continue the stored experiment")
  parser.add_argument("--result-directory",
    type=str,
    default=None,
    help="Path of the directory in which to save the experiment results (loss, cross-accuracy, ...) and checkpoints, empty for no saving")
  parser.add_argument("--evaluation-delta",
    type=int,
    default=100,
    help="How many training steps between model evaluations, 0 for no evaluation")
  parser.add_argument("--checkpoint-delta",
    type=int,
    default=0,
    help="How many training steps between experiment checkpointing, 0 or leave '--result-directory' empty for no checkpointing")
  parser.add_argument("--user-input-delta",
    type=int,
    default=0,
    help="How many training steps between two prompts for user command inputs, 0 for no user input")
  # Parse command line
  return parser.parse_args(sys.argv[1:])

with tools.Context("cmdline", "info"):
  args = process_commandline()
  # Parse additional arguments
  for name in ("init_multi", "init_mono", "gar", "attack", "model", "loss", "criterion"):
    name = f"{name}_args"
    keyval = getattr(args, name)
    setattr(args, name, dict() if keyval is None else tools.parse_keyval(keyval))
  # Count the number of real honest workers
  args.nb_honests = args.nb_workers - args.nb_real_byz
  if args.nb_honests < 0:
    tools.fatal(f"Invalid arguments: there are more real Byzantine workers ({args.nb_real_byz}) than total workers ({args.nb_workers})")
  # Check the learning rate and associated options
  if args.learning_rate_schedule is None:
    if args.learning_rate <= 0:
      tools.fatal(f"Invalid arguments: non-positive learning rate {args.learning_rate}")
    if args.learning_rate_decay < 0:
      tools.fatal(f"Invalid arguments: negative learning rate decay {args.learning_rate_decay}")
    if args.learning_rate_decay_delta <= 0:
      tools.fatal(f"Invalid arguments: non-positive learning rate decay delta {args.learning_rate_decay_delta}")
    config_learning_rate = (
      ("Initial", args.learning_rate),
      ("Half-decay", args.learning_rate_decay if args.learning_rate_decay > 0 else "none"),
      ("Update delta", args.learning_rate_decay_delta if args.learning_rate_decay > 0 else "n/a"))
    def compute_new_learning_rate(steps):
      if args.learning_rate_decay > 0 and steps % args.learning_rate_decay_delta == 0:
        return args.learning_rate / (steps / args.learning_rate_decay + 1)
  else:
    lr_schedule = tuple(float(number) if cnt % 2 == 0 else int(number) for cnt, number in enumerate(args.learning_rate_schedule.split(",")))
    def _transform_schedule(schedule):
      itr = iter(schedule)
      yield (0, next(itr))
      last = 0
      while True:
        try:
          step = next(itr)
        except StopIteration:
          return
        if step <= last:
          tools.fatal(f"Invalid arguments: learning rate schedule step numbers must by strictly increasing")
        yield (step, next(itr))
        last = step
    lr_schedule = tuple(_transform_schedule(lr_schedule))
    del _transform_schedule
    config_learning_rate = tuple((f"From step {step}", lr) for step, lr in lr_schedule)
    def compute_new_learning_rate(steps):
      for step, lr in lr_schedule:
        if steps == step:
          return lr
  # Check the momentum position
  momentum_at_values = ("update", "server", "worker")
  if args.momentum_at not in momentum_at_values:
    tools.fatal_unavailable(momentum_at_values, args.momentum_at, what="momentum position")
  # Check the number of local steps to perform per global step
  if args.nb_local_steps < 1:
    tools.fatal(f"Invalid arguments: non-positive number of local steps {args.nb_local_steps}")
  # Check no checkpoint to load if reproducibility requested
  if args.seed >= 0 and args.load_checkpoint is not None:
    tools.warning("Unable to enforce reproducibility when a checkpoint is loaded; ignoring seed")
    args.seed = -1
  # Check at least one gradient in past for studying purpose, or none if study disabled
  if args.result_directory is None:
    if args.nb_for_study > 0:
      args.nb_for_study = 0
    if args.nb_for_study_past > 0:
      args.nb_for_study_past = 0
  else:
    if args.nb_for_study_past < 1:
      tools.warning("At least one gradient must exist in the past to enable studying honest curvature; set '--nb-for-study-past 1'")
      args.nb_for_study_past = 1
    elif math.isclose(args.momentum, 0.0) and args.nb_for_study_past > 1:
      tools.warning("Momentum is (almost) zero, no need to store more than the previous honest gradient; set '--nb-for-study-past 1'")
      args.nb_for_study_past = 1
  # Print configuration
  def cmd_make_tree(subtree, level=0):
    if isinstance(subtree, tuple) and len(subtree) > 0 and isinstance(subtree[0], tuple) and len(subtree[0]) == 2:
      label_len = max(len(label) for label, _ in subtree)
      iterator  = subtree
    elif isinstance(subtree, dict):
      if len(subtree) == 0:
        return " - <none>"
      label_len = max(len(label) for label in subtree.keys())
      iterator  = subtree.items()
    else:
      return f" - {subtree}"
    level_spc = "  " * level
    res = ""
    for label, node in iterator:
      res += f"{os.linesep}{level_spc}· {label}{' ' * (label_len - len(label))}{cmd_make_tree(node, level + 1)}"
    return res
  if args.gars is None:
    cmdline_gars = (
      ("Name", args.gar),
      ("Arguments", args.gar_args))
  else:
    cmdline_gars = list()
    for info in args.gars.split(";"):
      info = info.split(",", maxsplit=2)
      if len(info) < 2:
        info.append("1")
      if len(info) < 3:
        info.append(None)
      else:
        try:
          info[2] = json.loads(info[2].strip())
        except json.decoder.JSONDecodeError:
          info[2] = "<parsing failed>"
      cmdline_gars.append((f"Frequency {info[1].strip()}", (
        ("Name", info[0].strip()),
        ("Arguments", info[2]))))
    cmdline_gars = tuple(cmdline_gars)
  cmdline_config = "Configuration" + cmd_make_tree((
    ("Reproducibility", "not enforced" if args.seed < 0 else (f"enforced (seed {args.seed})")),
    ("#workers", args.nb_workers),
    ("#local steps", "1 (standard)" if args.nb_local_steps == 1 else f"{args.nb_local_steps}"),
    ("#declared Byz.", args.nb_decl_byz),
    ("#actually Byz.", args.nb_real_byz),
    ("#study per step", "no study" if args.nb_for_study == 0 else max(args.nb_honests, args.nb_for_study)),
    ("#study for past", "no study" if args.nb_for_study_past == 0 else args.nb_for_study_past),
    ("Model", (
      ("Name", args.model),
      ("Arguments", args.model_args))),
    ("Initialization", (
      ("Mono", "<default>" if args.init_mono is None else args.init_mono),
      ("Arguments", args.init_mono_args),
      ("Multi", "<default>" if args.init_multi is None else args.init_multi),
      ("Arguments", args.init_multi_args))),
    ("Dataset", (
      ("Name", args.dataset),
      ("Batch size", (
        ("Training", args.batch_size),
        ("Testing", f"{args.batch_size_test} × {args.batch_size_test_reps}"))),
      ("Transforms", "none" if args.no_transform else "default"))),
    ("Loss", (
      ("Name", args.loss),
      ("Arguments", args.loss_args),
      ("Regularization", (
        ("l1", "none" if args.l1_regularize is None else f"{args.l1_regularize}"),
        ("l2", "none" if args.l2_regularize is None else f"{args.l2_regularize}"))))),
    ("Criterion", (
      ("Name", args.criterion),
      ("Arguments", args.criterion_args))),
    ("Optimizer", (
      ("Name", "sgd"),
      ("Learning rate", config_learning_rate),
      ("Momentum", (
        ("Type", "nesterov" if args.momentum_nesterov else "classical"),
        ("Where", f"at {args.momentum_at}"),
        ("Momentum", f"{args.momentum}"),
        ("Dampening", f"{args.dampening}"))),
      ("Weight decay", args.weight_decay),
      ("Gradient clip", "never" if args.gradient_clip is None else f"{args.gradient_clip}"))),
    ("Attack", (
      ("Name", args.attack),
      ("Arguments", args.attack_args))),
    ("Aggregation" if args.gars is None else "Aggregations", cmdline_gars)))
  print(cmdline_config)

# ---------------------------------------------------------------------------- #
# Setup
tools.success("Experiment setup...")

def result_make(name, *fields):
  """ Make and bind a new result file with a name, initialize with a header line.
  Args:
    name      Name of the result file
    fields... Name of each field, in order
  Raises:
    'KeyError' if name is already bound
    'RuntimeError' if no name can be bound
    Any exception that 'io.FileIO' can raise while opening/writing/flushing
  """
  # Check if results are to be output
  global args
  if args.result_directory is None:
    raise RuntimeError("No result is to be output")
  # Check if name is already bounds
  global result_fds
  if name in result_fds:
    raise KeyError(f"Name {name!r} is already bound to a result file")
  # Make the new file
  fd = (args.result_directory / name).open("w")
  fd.write("# " + ("\t").join(str(field) for field in fields))
  fd.flush()
  result_fds[name] = fd

def result_get(name):
  """ Get a valid descriptor to the bound result file, or 'None' if the given name is not bound.
  Args:
    name Given name
  Returns:
    Valid file descriptor, or 'None'
  """
  # Check if results are to be output
  global args
  if args.result_directory is None:
    return None
  # Return the bound descriptor, if any
  global result_fds
  return result_fds.get(name, None)

def result_store(fd, *entries):
  """ Store a line in a valid result file.
  Args:
    fd         Descriptor of the valid result file
    entries... Object(s) to convert to string and write in order in a new line
  """
  fd.write(os.linesep + ("\t").join(str(entry) for entry in entries))
  fd.flush()

with tools.Context("setup", "info"):
  # Enforce reproducibility if asked (see https://pytorch.org/docs/stable/notes/randomness.html)
  reproducible = args.seed >= 0
  if reproducible:
    torch.manual_seed(args.seed)
    import numpy
    numpy.random.seed(args.seed)
  torch.backends.cudnn.deterministic = reproducible
  torch.backends.cudnn.benchmark     = not reproducible
  # Configurations
  config = experiments.Configuration(dtype=torch.float32, device=(None if args.device.lower() == "auto" else args.device), noblock=True)
  if args.device_gar.lower() == "same":
    config_gar = config
  else:
    config_gar = experiments.Configuration(dtype=config["dtype"], device=(None if args.device_gar.lower() == "auto" else args.device_gar), noblock=config["non_blocking"])
  # Defense
  if args.gars is None:
    defense = aggregators.gars.get(args.gar)
    if defense is None:
      tools.fatal_unavailable(aggregators.gars, args.gar, what="aggregation rule")
  else:
    def generate_defense(gars):
      # Preprocess given configuration
      freq_sum = 0.
      defenses = list()
      for info in gars.split(";"):
        # Parse GAR info
        info = info.split(",", maxsplit=2)
        name = info[0].strip()
        if len(info) >= 2:
          freq = info[1].strip()
          if freq == "-":
            freq = 1.
          else:
            freq = float(freq)
        else:
          freq = 1.
        if len(info) >= 3:
          try:
            conf = json.loads(info[2].strip())
            if not isinstance(conf, dict):
              tools.fatal(f"Invalid GAR arguments for GAR {name!r}: expected a dictionary, got {getattr(type(conf), '__qualname__', '<unknown>')!r}")
          except json.decoder.JSONDecodeError as err:
            tools.fatal(f"Invalid GAR arguments for GAR {name!r}: {str(err).lower()}")
        else:
          conf = dict()
        # Recover association GAR function
        defense = aggregators.gars.get(name)
        if defense is None:
          tools.fatal_unavailable(aggregators.gars, name, what="aggregation rule")
        # Store parsed defense
        freq_sum += freq
        defenses.append((defense, freq_sum, conf))
      # Return closure
      def unchecked(**kwargs):
        sel = random.random() * freq_sum
        for func, freq, conf in defenses:
          if sel < freq:
            return func.unchecked(**kwargs, **conf)
        return func.unchecked(**kwargs, **conf)  # Gracefully handle numeric imprecision
      def check(**kwargs):
        for defense, _, conf in defenses:
          message = defense.check(**kwargs, **conf)
          if message is not None:
            return message
      return aggregators.make_gar(unchecked, check)
    defense = generate_defense(args.gars)
    args.gar_args = dict()
  # Attack
  attack = attacks.attacks.get(args.attack)
  if attack is None:
    tools.fatal_unavailable(attacks.attacks, args.attack, what="attack")
  # Model
  model = experiments.Model(args.model, config, init_multi=args.init_multi, init_multi_args=args.init_multi_args, init_mono=args.init_mono, init_mono_args=args.init_mono_args, **args.model_args)
  # Datasets
  if args.no_transform:
    train_transforms = test_transforms = torchvision.transforms.ToTensor()
  else:
    train_transforms = test_transforms = None # Let default values
  trainset, testset = experiments.make_datasets(args.dataset, args.batch_size, args.batch_size_test, train_transforms=train_transforms, test_transforms=test_transforms)
  model.default("trainset", trainset)
  model.default("testset", testset)
  # Losses
  loss = experiments.Loss(args.loss, **args.loss_args)
  if args.l1_regularize is not None:
    loss += args.l1_regularize * experiments.Loss("l1")
  if args.l2_regularize is not None:
    loss += args.l2_regularize * experiments.Loss("l2")
  criterion = experiments.Criterion(args.criterion, **args.criterion_args)
  model.default("loss", loss)
  model.default("criterion", criterion)
  # Optimizer
  # NOTE: The implementation of Nesterov in PyTorch may be incorrect, as it seems not to follow the definition in the original paper (http://www.cs.toronto.edu/~fritz/absps/momentum.pdf); so let's always keep 'nesterov=False' here
  optimizer = experiments.Optimizer("sgd", model, lr=args.learning_rate, momentum=0., dampening=0., weight_decay=args.weight_decay)
  model.default("optimizer", optimizer)
  # Miscellaneous storage (step counter, momentum gradients, ...)
  storage = experiments.Storage()
  # Make the result directory (if requested)
  if args.result_directory is not None:
    try:
      resdir = pathlib.Path(args.result_directory).resolve()
      resdir.mkdir(mode=0o755, parents=True, exist_ok=True)
      args.result_directory = resdir
    except Exception as err:
      tools.warning(f"Unable to create the result directory {str(resdir)!r} ({err}); no result will be stored")
    else:
      result_fds = dict()
      try:
        # Make evaluation file
        if args.evaluation_delta > 0:
          result_make("eval", "Step number", "Cross-accuracy")
        # Make study file
        if args.nb_for_study > 0:
          result_make("study", "Step number", "Training point count",
            "Average loss", "l2 from origin",
            "Sampled gradient deviation", "Honest gradient deviation", "Attack gradient deviation",
            "Sampled gradient norm", "Honest gradient norm", "Attack gradient norm", "Defense gradient norm",
            "Sampled max coordinate", "Honest max coordinate", "Attack max coordinate", "Defense max coordinate",
            "Sampled-honest cosine", "Sampled-attack cosine", "Sampled-defense cosine", "Honest-attack cosine", "Honest-defense cosine", "Attack-defense cosine",
            "Sampled-prev cosine", "Sampled composite curvature",
            "Attack acceptation ratio")
        # Store the configuration info and JSON representation
        (args.result_directory / "config").write_text(cmdline_config + os.linesep)
        with (args.result_directory / "config.json").open("w") as fd:
          def convert_to_supported_json_type(x):
            if type(x) in {str, int, float, bool, type(None), dict, list}:
              return x
            elif type(x) is set:
              return list(x)
            else:
              return str(x)
          datargs = dict((name, convert_to_supported_json_type(getattr(args, name))) for name in dir(args) if len(name) > 0 and name[0] != "_")
          del convert_to_supported_json_type
          json.dump(datargs, fd, ensure_ascii=False, indent="\t")
      except Exception as err:
        tools.warning(f"Unable to create some result files in directory {str(resdir)!r} ({err}); some result(s) may be missing")
  else:
    args.result_directory = None
    if args.checkpoint_delta != 0:
      args.checkpoint_delta = 0
      tools.warning("Argument '--checkpoint-delta' has been ignored as no '--result-directory' has been specified")

# ---------------------------------------------------------------------------- #
# Training
tools.success("Training...")

class CheckConvertTensorError(RuntimeError):
  pass

def check_convert_tensor(tensor, refshape, config=config, errname="tensor"):
  """ Assert the given parameter is a tensor of the given reference shape, then convert it to the current config.
  Args:
    tensor   Tensor instance to assert
    refshape Reference shape to match
    config   Target configuration for the tensor
    errname  Name of what the tensor represents (only for the error messages)
  Returns:
    Asserted and converted tensor
  Raises:
    'CheckConvertTensorError' with explanatory message
  """
  if not isinstance(tensor, torch.Tensor):
    raise CheckConvertTensorError(f"no/invalid {errname}")
  if tensor.shape != refshape:
    raise CheckConvertTensorError(f"{errname} has unexpected shape, expected {refshape}, got {tensor.shape}")
  try:
    return tensor.to(device=config["device"], dtype=config["dtype"], non_blocking=config["non_blocking"])
  except Exception as err:
    raise CheckConvertTensorError(f"converting/moving {errname} failed (err)")

# Load/initialize experiment
version = 4 # Must be unique and incremented on every change that makes previously created checkpoint incompatible
with tools.Context("load", "info"):
  if args.load_checkpoint is not None:
    try:
      experiments.Checkpoint().load(args.load_checkpoint).restore(model).restore(optimizer).restore(storage)
    except Exception as err:
      tools.fatal(f"Unable to load checkpoint {args.load_checkpoint!r}: {err}")
    # Check version
    stored_version = storage.get("version", None)
    if stored_version != version:
      tools.fatal(f"Unable to load checkpoint {args.load_checkpoint!r}: expected version {version!r}, got {stored_version!r}")
    # Check step counter
    steps = storage.get("steps", None)
    if not isinstance(steps, int) or steps < 0:
      tools.fatal(f"Unable to load checkpoint {args.load_checkpoint!r}: missing/invalid step counter, expected non-negative integer, got {steps!r}")
    # Check training point counter
    datapoints = storage.get("datapoints", None)
    if not isinstance(datapoints, int) or datapoints < 0:
      tools.fatal(f"Unable to load checkpoint {args.load_checkpoint!r}: missing/invalid training point counter, expected non-negative integer, got {datapoints!r}")
    # Check stored gradients (for momentum, if need be) and put them on the right device with the right dtype
    refshape = model.get().shape
    try:
      if args.momentum_at == "worker":
        grad_momentum_workers = storage.get("momentum", None)
        if grad_momentum_workers is None or not isinstance(grad_momentum_workers, list):
          tools.fatal(f"Unable to load checkpoint {args.load_checkpoint!r}: no/invalid stored momentum gradients")
        if len(grad_momentum_workers) < args.nb_honests:
          tools.fatal(f"Unable to load checkpoint {args.load_checkpoint!r}: not enough stored momentum gradients, expected {args.nb_honests}, got {len(grad_momentum_workers)}")
        elif len(grad_momentum_workers) > args.nb_honests:
          tools.warning(f"Found too many momentum gradients in checkpoint's storage {args.load_checkpoint!r}, expected {args.nb_honests}, got {len(grad_momentum_workers)}")
        for i, grad in enumerate(grad_momentum_workers):
          res = check_convert_tensor(grad, refshape, errname="stored momentum gradient")
          grad.data = res
      else:
        grad_momentum_server = check_convert_tensor(storage.get("momentum", None), refshape, errname="stored momentum gradient")
    except CheckConvertTensorError as err:
      tools.fatal(f"Unable to load checkpoint {args.load_checkpoint!r}: {err}")
    # Check original parameters
    if result_get("study") is not None:
      try:
        params_origin = check_convert_tensor(storage.get("origin", None), refshape, errname="stored original parameters")
      except CheckConvertTensorError as err:
        tools.fatal(f"Unable to load checkpoint {args.load_checkpoint!r}: {err}")
    else:
      if "origin" in storage:
        tools.warning(f"Found useless original parameters in checkpoint's storage {args.load_checkpoint!r}")
  else:
    # Initialize version
    storage["version"] = version
    # Initialize step and training point counters
    storage["steps"] = 0
    storage["datapoints"] = 0
    # Initialize stored gradients (for momentum, if need be)
    if args.momentum_at == "worker":
      grad_momentum_workers = storage["momentum"] = list(torch.zeros_like(model.get()) for _ in range(args.nb_honests))
    else:
      grad_momentum_server = storage["momentum"] = torch.zeros_like(model.get())
    # Initialize original parameters (if need be)
    if result_get("study") is not None:
      params_origin = storage["origin"] = model.get().clone().detach_()
# NOTE: 'args.load_checkpoint' is from this point on to be considered a flag: not None <=> a checkpoint has just been loaded

# Training until limit or stopped
with tools.Context("training", "info"):
  steps_limit  = None if args.nb_steps < 0 else storage["steps"] + args.nb_steps
  was_training = False
  current_lr   = None
  fd_eval      = result_get("eval")
  fd_study     = result_get("study")
  if fd_study is not None:
    # Make the collection of '--nb-for-study-past' past gradients ('PastGrad')
    PastGrad   = collections.namedtuple("PastGrad", ("grad", "norm"))
    grad_pasts = collections.deque(maxlen=args.nb_for_study_past)
  while not exit_is_requested():
    steps      = storage["steps"]
    datapoints = storage["datapoints"]
    # ------------------------------------------------------------------------ #
    # Evaluate if any milestone is reached
    milestone_evaluation = args.evaluation_delta > 0 and steps % args.evaluation_delta == 0
    milestone_checkpoint = args.checkpoint_delta > 0 and steps % args.checkpoint_delta == 0
    milestone_user_input = args.user_input_delta > 0 and steps % args.user_input_delta == 0
    milestone_any        = milestone_evaluation or milestone_checkpoint or milestone_user_input
    # Training notification (end)
    if milestone_any and was_training:
      print(" done.")
      was_training = False
    # Evaluation milestone reached
    if milestone_evaluation:
      print(f"Accuracy (step {steps})...", end="", flush=True)
      res = model.eval()
      for _ in range(args.batch_size_test_reps - 1):
        res += model.eval()
      acc = res[0].item() / res[1].item()
      print(f" {acc * 100.:.2f}%.")
      # Store the evaluation result
      if fd_eval is not None:
        result_store(fd_eval, steps, acc)
    # Saving milestone reached
    if milestone_checkpoint:
      if args.load_checkpoint is None: # Avoid overwriting the checkpoint we just loaded
        filename = args.result_directory / f"checkpoint-{steps}" # Result directory is set and valid at this point
        print(f"Saving in {filename.name!r}...", end="", flush=True)
        try:
          experiments.Checkpoint().snapshot(model).snapshot(optimizer).snapshot(storage).save(filename, overwrite=True)
          print(" done.")
        except:
          tools.warning(" fail.")
          with tools.Context("traceback", "trace"):
            traceback.print_exc()
    args.load_checkpoint = None
    # User input milestone
    if milestone_user_input:
      tools.interactive()
    # Check if reach step limit
    if steps_limit is not None and steps >= steps_limit:
      # Training notification (end)
      if was_training:
        print(" done.")
        was_training = False
      # Leave training loop
      break
    # Training notification (begin)
    if milestone_any and not was_training:
      print("Training...", end="", flush=True)
      was_training = True
    # Update learning rate
    new_lr = compute_new_learning_rate(steps)
    if new_lr is not None:
      optimizer.set_lr(new_lr)
      current_lr = new_lr
    # ------------------------------------------------------------------------ #
    # Compute honest gradients and losses (if it makes sense)
    grad_sampleds = list()
    if args.nb_local_steps == 1: # Standard SGD (fast path: avoid taking a deep snapshot)
      loss_sampleds = list()
      if args.momentum_nesterov:
        # Snapshot the model (we are going to move it)
        local_chckpt = experiments.Checkpoint().snapshot(model, deepcopy=True)
        # Move parameters following the server-side momentum (if need be)
        if args.momentum_at != "worker":
          model.get().sub_(grad_momentum_server, alpha=(args.momentum * current_lr))
        # For each worker (or enough to meet study requirements)
        for i in range(max(args.nb_honests, args.nb_for_study)):
          # Move parameters following the worker-side momentum (if need be)
          if args.momentum_at == "worker":
            model.get().sub_(grad_momentum_workers[i], alpha=(args.momentum * current_lr))
          # Compute gradient
          grad, loss = model.backprop(outloss=True)
          # Restore parameters (worker-side momentum)
          if args.momentum_at == "worker":
            local_chckpt.restore(model)
          # Loss append
          loss_sampleds.append(loss.clone().detach_().item())
          # Gradient clip and append
          if args.gradient_clip is not None:
            grad_norm = grad.norm().item()
            if grad_norm > args.gradient_clip:
              grad.mul_(args.gradient_clip / grad_norm)
          grad_sampleds.append(grad.clone().detach_())
        # Restore parameters (server-side momentum)
        if args.momentum_at != "worker":
          local_chckpt.restore(model)
      else:
        # For each worker (or enough to meet study requirements)
        for _ in range(max(args.nb_honests, args.nb_for_study)):
          grad, loss = model.backprop(outloss=True)
          # Loss append
          loss_sampleds.append(loss.clone().detach_().item())
          # Gradient clip and append
          if args.gradient_clip is not None:
            grad_norm = grad.norm().item()
            if grad_norm > args.gradient_clip:
              grad.mul_(args.gradient_clip / grad_norm)
          grad_sampleds.append(grad.clone().detach_())
    else: # Multi-steps SGD
      # NOTE: See previous version to do code review
      tools.fatal("Multi-steps SGD disabled until code review")
    # Select honest gradients, applying momentum before aggregating (if need be)
    if args.momentum_at == "worker":
      grad_honests = list()
      for gmtm, grad in zip(grad_momentum_workers, grad_sampleds[:args.nb_honests]):
        gmtm.mul_(args.momentum).add_(grad, alpha=(1. - args.dampening))
        grad_honests.append(gmtm)
    elif args.momentum_at == "server":
      grad_honests = list()
      for grad in grad_sampleds[:args.nb_honests]:
        grad_honests.append(grad.mul(1. - args.dampening).add_(grad_momentum_server, alpha=args.momentum))
    else:
      grad_honests = grad_sampleds[:args.nb_honests]
    # Move the honest gradients to the GAR device
    if config_gar is not config:
      grad_honests_gar = list(grad.to(device=config_gar["device"], non_blocking=config_gar["non_blocking"]) for grad in grad_honests)
    else:
      grad_honests_gar = grad_honests
    # ------------------------------------------------------------------------ #
    # Compute the Byzantine gradients (here as the adversary knows the momentum)
    grad_attacks = attack.checked(grad_honests=grad_honests_gar, f_decl=args.nb_decl_byz, f_real=args.nb_real_byz, model=model, defense=defense, **args.attack_args)
    # ------------------------------------------------------------------------ #
    # Aggregate and update the model
    grad_defense = defense.checked(gradients=(grad_honests_gar + grad_attacks), f=args.nb_decl_byz, model=model, **args.gar_args)
    accept_ratio = math.nan if defense.influence is None else defense.influence(grad_honests_gar, grad_attacks, f=args.nb_decl_byz, **args.gar_args)
    # Move the defense gradient back to the main device
    if config_gar is not config:
      for grad in grad_attacks:
        grad.data = grad.to(device=config["device"], non_blocking=config["non_blocking"])
      grad_defense = grad_defense.to(device=config["device"], non_blocking=config["non_blocking"])
    # Compute l2-distance from origin (if needed for study)
    if fd_study is not None:
      l2_origin = model.get().sub(params_origin).norm().item()
    # Model update (this code handles momentum computation, not PyTorch)
    if args.momentum_at == "worker":
      model.update(grad_defense)
    elif args.momentum_at == "server":
      grad_momentum_server = storage["momentum"] = grad_defense  # No need to clone (not updated)
      model.update(grad_defense)
    elif args.momentum_at == "update":
      grad_momentum_server.mul_(args.momentum).add_(grad_defense, alpha=(1. - args.dampening))
      model.update(grad_momentum_server)
    # ------------------------------------------------------------------------ #
    # Store study (if requested)
    if fd_study is not None:
      # Compute average loss (if 'loss_sampled is not None', 'len(loss_sampleds) > 0' is guaranteed)
      loss_avg = math.nan if loss_sampleds is None else sum(loss_sampleds) / len(loss_sampleds)
      # Compute the sampled and honest gradients norm average, norm deviation and max absolute coordinate
      sampled_grad_avg, sampled_norm_avg, sampled_norm_dev, sampled_norm_max = tools.compute_avg_dev_max(grad_sampleds)
      honest_grad_avg,  honest_norm_avg,  honest_norm_dev,  honest_norm_max  = tools.compute_avg_dev_max(grad_honests)
      attack_grad_avg,  attack_norm_avg,  attack_norm_dev,  attack_norm_max  = tools.compute_avg_dev_max(grad_attacks)
      # Compute the defense norm average and max absolute coordinate
      defense_grad = grad_defense # (Mere renaming for consistency)
      defense_norm_avg = defense_grad.norm().item()
      defense_norm_max = defense_grad.abs().max().item()
      # Compute cosine of solid angles
      cosin_splhon = torch.dot(sampled_grad_avg, honest_grad_avg).div_(sampled_norm_avg).div_(honest_norm_avg).item()
      cosin_splatt = math.nan if attack_grad_avg is None else torch.dot(sampled_grad_avg, attack_grad_avg).div_(sampled_norm_avg).div_(attack_norm_avg).item()
      cosin_spldef = torch.dot(sampled_grad_avg, defense_grad).div_(sampled_norm_avg).div_(defense_norm_avg).item()
      cosin_honatt = math.nan if attack_grad_avg is None else torch.dot(honest_grad_avg, attack_grad_avg).div_(honest_norm_avg).div_(attack_norm_avg).item()
      cosin_hondef = torch.dot(honest_grad_avg, defense_grad).div_(honest_norm_avg).div_(defense_norm_avg).item()
      cosin_attdef = math.nan if attack_grad_avg is None else torch.dot(attack_grad_avg, defense_grad).div_(attack_norm_avg).div_(defense_norm_avg).item()
      # Compute past sampled solid angle and curvature
      if len(grad_pasts) > 0:
        cosin_sampled = torch.dot(sampled_grad_avg, grad_pasts[0].grad).div_(sampled_norm_avg).div_(grad_pasts[0].norm).item()
        curv_sampled  = args.momentum * sum((args.momentum ** i * torch.dot(sampled_grad_avg, grad_past.grad).item()) for i, grad_past in enumerate(grad_pasts))
      else:
        cosin_sampled = math.nan
        curv_sampled  = math.nan
      # Store the new past gradient (automatic rolling)
      grad_pasts.appendleft(PastGrad(sampled_grad_avg, sampled_norm_avg))
      # Store the result (float-to-string format chosen so not to lose precision)
      float_format = {torch.float16: "%.4e", torch.float32: "%.8e", torch.float64: "%.16e"}.get(config["dtype"], "%s")
      result_store(fd_study, steps, datapoints,
        float_format % loss_avg, float_format % l2_origin,
        float_format % sampled_norm_dev, float_format % honest_norm_dev, float_format % attack_norm_dev,
        float_format % sampled_norm_avg, float_format % honest_norm_avg, float_format % attack_norm_avg, float_format % defense_norm_avg,
        float_format % sampled_norm_max, float_format % honest_norm_max, float_format % attack_norm_max, float_format % defense_norm_max,
        float_format % cosin_splhon, float_format % cosin_splatt, float_format % cosin_spldef, float_format % cosin_honatt, float_format % cosin_hondef, float_format % cosin_attdef,
        float_format % cosin_sampled, float_format % curv_sampled,
        accept_ratio)
    # ------------------------------------------------------------------------ #
    # Increase the step counter
    storage["steps"]      = steps + 1
    storage["datapoints"] = datapoints + args.batch_size * args.nb_honests * args.nb_local_steps
  # Training notification (end)
  if was_training:
    print(" interrupted.")

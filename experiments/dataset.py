# coding: utf-8
###
 # @file   dataset.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Dataset wrappers/helpers.
###

__all__ = ["get_default_transform", "Dataset", "make_sampler", "make_datasets",
           "batch_dataset"]

import tools

import pathlib
import random
import tempfile
import torch
import torchvision
import types

# ---------------------------------------------------------------------------- #
# Default image transformations

# Collection of default transforms, <dataset name> -> (<train transforms>, <test transforms>)
transforms_horizontalflip = [
  torchvision.transforms.RandomHorizontalFlip(),
  torchvision.transforms.ToTensor()]
transforms_mnist = [
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize((0.1307,), (0.3081,))] # Transforms from "A Little is Enough" (https://github.com/moranant/attacking_distributed_learning)
transforms_cifar = [
  torchvision.transforms.RandomHorizontalFlip(),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] # Transforms from https://github.com/kuangliu/pytorch-cifar

# Per-dataset image transformations (automatically completed, see 'Dataset._get_datasets')
transforms = {
  "mnist":        (transforms_mnist, transforms_mnist),
  "fashionmnist": (transforms_horizontalflip, transforms_horizontalflip),
  "cifar10":      (transforms_cifar, transforms_cifar),
  "cifar100":     (transforms_cifar, transforms_cifar),
  "imagenet":     (transforms_horizontalflip, transforms_horizontalflip) }

def get_default_transform(dataset, train):
  """ Get the default transform associated with the given dataset name.
  Args:
    dataset Case-sensitive dataset name, or None to get no transformation
    train   Whether the transformation is for the training set (always ignored if None is given for 'dataset')
  Returns:
    Associated default transformations (always exist)
  """
  global transforms
  # Fetch transformation
  transform = transforms.get(dataset)
  # Not found (not a torchvision dataset)
  if transform is None:
    return None
  # Return associated transform
  return torchvision.transforms.Compose(transform[0 if train else 1])

# ---------------------------------------------------------------------------- #
# Dataset loader-batch producer wrapper class

class Dataset:
  """ Dataset wrapper class.
  """

  # Default dataset root directory path
  __default_root = None

  @classmethod
  def get_default_root(self):
    """ Lazy-initialize and return the default dataset root directory path.
    Returns:
      '__default_root'
    """
    # Fast-path already loaded
    if self.__default_root is not None:
      return self.__default_root
    # Generate the default path
    self.__default_root = pathlib.Path(__file__).parent / "datasets" / "cache"
    # Warn if the path does not exist and fallback to '/tmp'
    if not self.__default_root.exists():
      tmpdir = tempfile.gettempdir()
      tools.warning(f"Default dataset root {str(self.__default_root)!r} does not exist, falling back to local temporary directory {tmpdir!r}", context="experiments")
      self.__default_root = pathlib.Path(tmpdir)
    # Return the path
    return self.__default_root

  # Map 'lower-case names' -> 'dataset class' available in PyTorch
  __datasets = None

  @classmethod
  def _get_datasets(self):
    """ Lazy-initialize and return the map '__datasets'.
    Returns:
      '__datasets'
    """
    global transforms
    # Fast-path already loaded
    if self.__datasets is not None:
      return self.__datasets
    # Initialize the dictionary
    self.__datasets = dict()
    # Populate this dictionary with TorchVision's datasets
    for name in dir(torchvision.datasets):
      if len(name) == 0 or name[0] == "_": # Ignore "protected" members
        continue
      constructor = getattr(torchvision.datasets, name)
      if isinstance(constructor, type): # Heuristic
        def make_builder(constructor, name):
          def builder(root, batch_size=None, shuffle=False, num_workers=1, *args, **kwargs):
            # Try to build the dataset instance
            data = constructor(root, *args, **kwargs)
            assert isinstance(data, torch.utils.data.Dataset), f"Internal heuristic failed: {name!r} was not a dataset name"
            # Ensure there is at least a tensor transformation for each torchvision dataset
            if name not in transforms:
              transforms[name] = torchvision.transforms.ToTensor()
            # Wrap into a loader
            batch_size = batch_size or len(data)
            loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            # Wrap into an infinite batch sampler generator
            return make_sampler(loader)
          return builder
        self.__datasets[name.lower()] = make_builder(constructor, name)
    # Dynamically add the custom datasets from subdirectory 'datasets/'
    def add_custom_datasets(name, module, _):
      nonlocal self
      # Check if has exports, fallback otherwise
      exports = getattr(module, "__all__", None)
      if exports is None:
        tools.warning(f"Dataset module {name!r} does not provide '__all__'; falling back to '__dict__' for name discovery")
        exports = (name for name in dir(module) if len(name) > 0 and name[0] != "_")
      # Register the association 'name -> constructor' for all the datasets
      exported = False
      for dataset in exports:
        # Check dataset name type
        if not isinstance(dataset, str):
          tools.warning(f"Dataset module {name!r} exports non-string name {dataset!r}; ignored")
          continue
        # Recover instance from name
        constructor = getattr(module, dataset, None)
        # Check instance is callable (it's only an heuristic...)
        if not callable(constructor):
          continue
        # Register callable with composite name
        exported = True
        fullname = f"{name}-{dataset}"
        if fullname in self.__datasets:
          tools.warning(f"Unable to make available dataset {dataset!r} from module {name!r}, as the name {fullname!r} already exists")
          continue
        self.__datasets[fullname] = constructor
      if not exported:
        tools.warning(f"Dataset module {name!r} does not export any valid constructor name through '__all__'")
    with tools.Context("datasets", None):
      tools.import_directory(pathlib.Path(__file__).parent / "datasets", {"__package__": f"{__package__}.datasets"}, post=add_custom_datasets)
    # Return the dictionary
    return self.__datasets

  def __init__(self, data, name=None, root=None, *args, **kwargs):
    """ Dataset builder constructor.
    Args:
      data Dataset string name, (infinite) generator instance (that will be used to generate samples), or any other instance (that will then be fed as the only sample)
      name Optional user-defined dataset name, to attach to some error messages for debugging purpose
      root Dataset cache root directory to use, None for default (only relevant if 'data' is a dataset name)
      ...  Forwarded (keyword-)arguments to the dataset constructor, ignored if 'data' is not a string
    Raises:
      'TypeError' if the some of the given (keyword-)arguments cannot be used to call the dataset or loader constructor or the batch loader
    """
    # Handle different dataset types
    if isinstance(data, str): # Load sampler from available datasets
      if name is None:
        name = data
      datasets = type(self)._get_datasets()
      build = datasets.get(name, None)
      if build is None:
        raise tools.UnavailableException(datasets, name, what="dataset name")
      root = root or type(self).get_default_root()
      self._iter = build(root=root, *args, **kwargs)
    elif isinstance(data, types.GeneratorType): # Forward sampling to custom generator
      if name is None:
        name = "<generator>"
      self._iter = data
    else: # Single-batch dataset of any value
      if name is None:
        name = "<single-batch>"
      def single_batch():
        while True:
          yield data
      self._iter = single_batch()
    # Finalization
    self.name = name

  def __str__(self):
    """ Compute the "informal", nicely printable string representation of this dataset.
    Returns:
      Nicely printable string
    """
    return f"dataset {self.name}"

  def sample(self, config=None):
    """ Sample the next batch from this dataset.
    Args:
      config Target configuration for the sampled tensors
    Returns:
      Next batch
    """
    tns = next(self._iter)
    if config is not None:
      tns = type(tns)(tn.to(device=config["device"], non_blocking=config["non_blocking"]) for tn in tns)
    return tns

  def epoch(self, config=None):
    """ Return a full epoch iterable from this dataset.
    Args:
      config Target configuration for the sampled tensors
    Returns:
      Full epoch iterable
    Notes:
      Only work for dataset based on PyTorch's DataLoader
    """
    # Assert dataset based on DataLoader
    assert isinstance(self._loader, torch.utils.data.DataLoader), "Full epoch iteration only possible for PyTorch's DataLoader-based datasets"
    # Return a full epoch iterator
    epoch = self._loader.__iter__()
    def generator():
      nonlocal epoch
      try:
        while True:
          tns = next(epoch)
          if config is not None:
            tns = type(tns)(tn.to(device=config["device"], non_blocking=config["non_blocking"]) for tn in tns)
          yield tns
      except StopIteration:
        return
    return generator()

# ---------------------------------------------------------------------------- #
# Dataset helpers

def make_sampler(loader):
  """ Infinite sampler generator from a dataset loader.
  Args:
    loader Dataset loader to use
  Yields:
    Sample, forever (transparently iterating the given loader again and again)
  """
  itr = None
  while True:
    for _ in range(2):
      # Try sampling the next batch
      if itr is not None:
        try:
          yield next(itr)
          break
        except StopIteration:
          pass
      # Ask loader for a new iteration
      itr = iter(loader)
    else:
      raise RuntimeError(f"Unable to sample a new batch from dataset {name!r}")

def make_datasets(dataset, train_batch=None, test_batch=None, train_transforms=None, test_transforms=None, num_workers=1, **custom_args):
  """ Helper to make new instances of training and testing datasets.
  Args:
    dataset          Case-sensitive dataset name
    train_batch      Training batch size, None or 0 for maximum possible
    test_batch       Testing batch size, None or 0 for maximum possible
    train_transforms Transformations to apply on the training set, None for default for the given dataset
    test_transforms  Transformations to apply on the testing set, None for default for the given dataset
    num_workers      Positive number of workers for each of the training and testing datasets, or tuple for each of them
    ...              Additional dataset-dependent keyword-arguments
  Returns:
    Training dataset, testing dataset
  """
  # Pre-process arguments
  train_transforms = train_transforms or get_default_transform(dataset, True)
  test_transforms = test_transforms or get_default_transform(dataset, False)
  num_workers_errmsg = "Expected either a positive int or a tuple of 2 positive ints for parameter 'num_workers'"
  if isinstance(num_workers, int):
    assert num_workers > 0, num_workers_errmsg
    train_workers = test_workers = num_workers
  else:
    assert isinstance(num_workers, tuple) and len(num_workers) == 2, num_workers_errmsg
    train_workers, test_workers = num_workers
    assert isinstance(train_workers, int) and train_workers > 0, num_workers_errmsg
    assert isinstance(test_workers, int)  and test_workers > 0,  num_workers_errmsg
  # Make the datasets
  trainset = Dataset(dataset, train=True, download=True, batch_size=train_batch,
      shuffle=True, num_workers=train_workers, transform=train_transforms, **custom_args)
  testset = Dataset(dataset, train=False, download=False, batch_size=test_batch,
      shuffle=False, num_workers=test_workers, transform=test_transforms, **custom_args)
  # Return the datasets
  return trainset, testset

def batch_dataset(inputs, labels, train=False, batch_size=None, split=0.75):
  """ Batch a given raw (tensor) dataset into either a training or testing infinite sampler generators.
  Args:
    inputs     Tensor of positive dimension containing input data
    labels     Tensor of same shape as 'inputs' containing expected output data
    train      Whether this is for training (basically adds shuffling)
    batch_size Training batch size, None (or 0) for maximum batch size
    split      Fraction of datapoints to use in the train set if < 1, or #samples in the train set if ≥ 1
  Returns:
    Training or testing set infinite sampler generator (with uniformly sampled batches),
    Test set infinite sampler generator (without random sampling)
  """
  def train_gen(inputs, labels, batch):
    cursor = 0
    datalen = len(inputs)
    shuffle = list(range(datalen))
    random.shuffle(shuffle)
    while True:
      end = cursor + batch
      if end > datalen:
        select = shuffle[cursor:]
        random.shuffle(shuffle)
        select += shuffle[:(end % datalen)]
      else:
        select = shuffle[cursor:end]
      yield inputs[select], labels[select]
      cursor = end % datalen
  def test_gen(inputs, labels, batch):
    cursor = 0
    datalen = len(inputs)
    while True:
      end = cursor + batch
      if end > datalen:
        select = list(range(cursor, datalen)) + list(range(end % datalen))
        yield inputs[select], labels[select]
      else:
        yield inputs[cursor:end], labels[cursor:end]
      cursor = end % datalen
  # Split dataset
  dataset_len = len(inputs)
  if dataset_len < 1 or len(labels) != dataset_len:
    raise RuntimeError(f"Invalid or different input/output tensor lengths, got {len(inputs)} for inputs, got {len(labels)} for labels")
  split_pos = min(max(1, int(dataset_len * split)) if split < 1 else split, dataset_len - 1)
  # Make and return generator according to flavor
  if train:
    train_len = split_pos
    batch_size = min(batch_size or train_len, train_len)
    return train_gen(inputs[:split_pos], labels[:split_pos], batch_size)
  else:
    test_len = dataset_len - split_pos
    batch_size = min(batch_size or test_len, test_len)
    return test_gen(inputs[split_pos:], labels[split_pos:], batch_size)

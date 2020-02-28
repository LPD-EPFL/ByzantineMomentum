# coding: utf-8
###
 # @file   dataset.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2020 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Dataset wrappers/helpers.
###

__all__ = ["get_default_transform", "Dataset", "make_datasets"]

import tools

import pathlib
import tempfile
import torch
import torchvision
import types

# ---------------------------------------------------------------------------- #
# Default transformations

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
  # Fast path not found
  if transform is None:
    return torchvision.transforms.ToTensor()
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
    self.__default_root = pathlib.Path(__file__).parent / "datasets"
    # Warn if the path does not exist and fallback to '/tmp'
    if not self.__default_root.exists():
      tmpdir = tempfile.gettempdir()
      tools.warning("Default dataset root %r does not exist, falling back to local temporary directory %r" % (str(self.__default_root), tmpdir), context="experiments")
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
    # Fast-path already loaded
    if self.__datasets is not None:
      return self.__datasets
    # Initialize the dictionary
    self.__datasets = dict()
    # Simply populate this dictionary
    for name in dir(torchvision.datasets):
      if len(name) == 0 or name[0] == "_": # Ignore "protected" members
        continue
      builder = getattr(torchvision.datasets, name)
      if isinstance(builder, type): # Heuristic
        self.__datasets[name.lower()] = builder
    # Return the dictionary
    return self.__datasets

  def __init__(self, data, name=None, ds_args=(), ds_kwargs={}, ld_args=(), ld_kwargs={}):
    """ Dataset builder constructor.
    Args:
      data       Dataset string name, 'torch.utils.data.Dataset' instance, 'torch.utils.data.DataLoader' instance, or any other instance (that will then be fed as the only batch)
      name       Optional user-defined dataset name, to attach to some error messages for debugging purpose
      ds_args    Arguments forwarded to the dataset constructor, ignored if 'name_ds_ld' is not a string
      ds_kwargs  Keyword-arguments forwarded to the dataset constructor, ignored if 'name_ds_ld' is not a string
      ld_args    Arguments forwarded to the loader constructor, ignored if 'name_ds_ld' is not a string or a Dataset instance
      ld_kwargs  Keyword-arguments forwarded to the loader constructor, ignored if 'name_ds_ld' is not a string or a Dataset instance
    Raises:
      'TypeError' if the some of the given (keyword-)arguments cannot be used to call the dataset or loader constructor or the batch loader
    """
    # Pre-handling instantiate dataset from name
    if isinstance(data, str):
      if name is None:
        name = data
      datasets = type(self)._get_datasets()
      build = datasets.get(name, None)
      if build is None:
        raise tools.UnavailableException(datasets, name, what="dataset name")
      data = build(*ds_args, **ds_kwargs)
      assert isinstance(data, torch.utils.data.Dataset), "Internal heuristic failed: %r was not a dataset name" % data
    # Pre-handling instantiate dataset loader
    if isinstance(data, torch.utils.data.Dataset):
      self.dataset = data
      data = torch.utils.data.DataLoader(data, *ld_args, **ld_kwargs)
    else:
      self.dataset = None
    # Handle different dataset types
    if isinstance(data, torch.utils.data.DataLoader): # Data loader for sampling
      if name is None:
        name = "<custom loader>"
      self._loader = data
      self._iter   = None
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
    return "dataset %s" % self.name

  def sample(self, config=None):
    """ Sample the next batch from this dataset.
    Args:
      config Target configuration for the sampled tensors
    Returns:
      Next batch
    """
    for _ in range(2):
      # Try sampling a batch
      if self._iter is not None:
        try:
          tns = next(self._iter)
          if config is not None:
            tns = type(tns)(tn.to(device=config["device"], non_blocking=config["non_blocking"]) for tn in tns)
          return tns
        except StopIteration:
          pass
      # Ask loader (if any) for a new iteration
      loader = getattr(self, "_loader", None)
      if loader is not None:
        self._iter = loader.__iter__()
    raise RuntimeError("Unable to sample a new batch from dataset %r" % self.name)

def make_datasets(dataset, train_batch, test_batch, train_transforms=None, test_transforms=None, num_workers=1):
  """ Helper to make new instances of training and testing datasets.
  Args:
    dataset          Case-sensitive dataset name
    train_batch      Training batch size
    test_batch       Testing batch size
    train_transforms Transformations to apply on the training set, None for default for the given dataset
    test_transforms  Transformations to apply on the testing set, None for default for the given dataset
    num_workers      Positive number of workers for each of the training and testing datasets, or tuple for each of them
  """
  # Pre-process arguments
  path_root = Dataset.get_default_root()
  if train_transforms is None:
    train_transforms = get_default_transform(dataset, True)
  if test_transforms is None:
    test_transforms = get_default_transform(dataset, False)
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
  trainset = Dataset(dataset,
    ds_kwargs={
      "root": path_root,
      "train": True,
      "download": True,
      "transform": train_transforms },
    ld_kwargs={
      "batch_size": train_batch,
      "shuffle": True,
      "num_workers": train_workers })
  testset  = Dataset(dataset,
    ds_kwargs={
      "root": path_root,
      "train": False,
      "download": False,
      "transform": test_transforms },
    ld_kwargs={
      "batch_size": test_batch,
      "shuffle": True,
      "num_workers": test_workers })
  # Return the datasets
  return trainset, testset

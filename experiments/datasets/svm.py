# coding: utf-8
###
 # @file   svm.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Lazy-(down)load and pre-process datasets from LIBSVM.
 # Website: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
###

__all__ = ["phishing"]

import requests
import torch

import experiments
import tools

# ---------------------------------------------------------------------------- #
# Configuration

# Default raw dataset URLs
default_url_phishing = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing"

# Default cache root directory
default_root=experiments.dataset.Dataset.get_default_root()

# ---------------------------------------------------------------------------- #
# Dataset lazy-loaders

# Raw phishing dataset
raw_phishing = None

def get_phishing(root, url):
  """ Lazy-load the phishing dataset.
  Args:
    root Dataset cache root directory
    url  URL to fetch raw dataset from, if not already in cache (None for no download)
  Returns:
    Input tensor,
    Label tensor
  """
  global raw_phishing
  const_filename = "phishing.pt"
  const_features = 68
  const_datatype = torch.float32
  # Fast path: return loaded dataset
  if raw_phishing is not None:
    return raw_phishing
  # Make dataset path
  dataset_file = root / const_filename
  # Fast path: pre-processed dataset already locally available
  if dataset_file.exists():
    with dataset_file.open("rb") as fd:
      # Load, lazy-store and return dataset
      dataset = torch.load(fd)
      raw_phishing = dataset
      return dataset
  elif url is None:
    raise RuntimeError("Phishing dataset not in cache and download disabled")
  # Download dataset
  tools.info("Downloading dataset...", end="", flush=True)
  try:
    response = requests.get(url)
  except Exception as err:
    tools.warning(" fail.")
    raise RuntimeError(f"Unable to get dataset (at {url}): {err}")
  tools.info(" done.")
  if response.status_code != 200:
    raise RuntimeError(f"Unable to fetch raw dataset (at {url}): GET status code {response.status_code}")
  # Pre-process dataset
  tools.info("Pre-processing dataset...", end="", flush=True)
  entries = response.text.strip().split("\n")
  inputs = torch.zeros(len(entries), const_features, dtype=const_datatype)
  labels = torch.empty(len(entries), dtype=const_datatype)
  for index, entry in enumerate(entries):
    entry = entry.split(" ")
    # Set label
    labels[index] = 1 if entry[0] == "1" else 0
    # Set input
    line = inputs[index]
    for pos, setter in enumerate(entry[1:]):
      try:
        offset, value = setter.split(":")
        line[int(offset) - 1] = float(value)
      except Exception as err:
        tools.warning(" fail.")
        raise RuntimeError(f"Unable to parse dataset (line {index + 1}, position {pos + 1}): {err}")
  labels.unsqueeze_(1)
  tools.info(" done.")
  # (Try to) save pre-processed dataset
  try:
    with dataset_file.open("wb") as fd:
      torch.save((inputs, labels), fd)
  except Exception as err:
    tools.warning(f"Unable to save pre-processed dataset: {err}")
  # Lazy-store and return dataset
  dataset = (inputs, labels)
  raw_phishing = dataset
  return dataset

# ---------------------------------------------------------------------------- #
# Dataset generators

def phishing(train=True, batch_size=None, root=None, download=False, *args, **kwargs):
  """ Phishing dataset generator builder.
  Args:
    train      Whether to get the training slice of the dataset
    batch_size Batch size (None or 0 for all in one single batch)
    root       Dataset cache root directory (None for default)
    download   Whether to allow to download the dataset if not cached locally
    ...        Ignored supplementary (keyword-)arguments
  Returns:
    Associated ataset generator
  """
  with tools.Context("phishing", None):
    # Get the raw dataset
    inputs, labels = get_phishing(root or default_root, None if download is None else default_url_phishing)
    # Make and return the associated generator
    return experiments.batch_dataset(inputs, labels, train, batch_size, split=8400)  # 8400 = 2⁴ × 3 × 5² × 7 (should help with divisibility)

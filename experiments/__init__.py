# coding: utf-8
###
 # @file   __init__.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Dataset/model/... wrappers/helpers, for more convenient gradient extraction and operations.
 # Heavily relies on the module 'torchvision'.
###

import pathlib

import tools

# ---------------------------------------------------------------------------- #
# Load all local modules

with tools.Context("experiments", None):
  tools.import_directory(pathlib.Path(__file__).parent, globals())

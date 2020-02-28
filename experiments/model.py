# coding: utf-8
###
 # @file   model.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2020 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Model wrappers/helpers.
###

__all__ = ["Model"]

import tools

import pathlib
import torch
import torchvision
import types

from .configuration import Configuration

# ---------------------------------------------------------------------------- #
# Model wrapper class

class Model:
  """ Model wrapper class.
  """

  # Map 'lower-case names' -> 'model constructor' available in PyTorch
  __models = None

  @classmethod
  def _get_models(self):
    """ Lazy-initialize and return the map '__models'.
    Returns:
      '__models'
    """
    # Fast-path already loaded
    if self.__models is not None:
      return self.__models
    # Initialize the dictionary
    self.__models = dict()
    # Populate this dictionary with TorchVision's models
    for name in dir(torchvision.models):
      if len(name) == 0 or name[0] == "_": # Ignore "protected" members
        continue
      builder = getattr(torchvision.models, name)
      if isinstance(builder, types.FunctionType): # Heuristic
        self.__models["torchvision-%s" % name.lower()] = builder
    # Dynamically add the custom models from subdirectory 'models/'
    def add_custom_models(name, module, _):
      nonlocal self
      # Check if has exports, fallback otherwise
      exports = getattr(module, "__all__", None)
      if exports is None:
        tools.warning("Model module %r does not provide '__all__'; falling back to '__dict__' for name discovery" % name)
        exports = (name for name in dir(module) if len(name) > 0 and name[0] != "_")
      # Register the association 'name -> constructor' for all the models
      exported = False
      for model in exports:
        # Check model name type
        if not isinstance(model, str):
          tools.warning("Model module %r exports non-string name %r; ignored" % (name, model))
          continue
        # Recover instance from name
        constructor = getattr(module, model, None)
        # Check instance is callable (it's only an heuristic...)
        if not callable(constructor):
          continue
        # Register callable with composite name
        exported = True
        fullname = "%s-%s" % (name, model)
        if fullname in self.__models:
          tools.warning("Unable to make available model %r from module %r, as the name %r already exists" % (model, name, fullname))
          continue
        self.__models[fullname] = constructor
      if not exported:
        tools.warning("Model module %r does not export any valid constructor name through '__all__'" % name)
    with tools.Context("models", None):
      tools.import_directory(pathlib.Path(__file__).parent / "models", {"__package__": "%s.models" % __package__}, post=add_custom_models)
    # Return the dictionary
    return self.__models

  def __init__(self, name_build, config=Configuration(), *args, **kwargs):
    """ Model builder constructor.
    Args:
      name_build Model name or constructor function
      config     Configuration to use for the parameter tensors
      ...        Additional (keyword-)arguments forwarded to the constructor
    Notes:
      If possible, data parallelism is enabled automatically
    """
    # Recover name/constructor
    if callable(name_build):
      name  = tools.fullqual(name_build)
      build = name_build
    else:
      models = type(self)._get_models()
      name  = str(name_build)
      build = models.get(name, None)
      if build is None:
        raise tools.UnavailableException(models, name, what="model name")
    # Build model
    with torch.no_grad():
      model = build(*args, **kwargs)
      if not isinstance(model, torch.nn.Module):
        raise tools.UserException("Expected built model %r to be an instance of 'torch.nn.Module', found %r instead" % (name, getattr(type(model), "__name__", "<unknown>")))
      model = model.to(**config)
      device = config["device"]
      if device.type == "cuda" and device.index is None: # Model is on GPU and not explicitly restricted to one particular card => enable data parallelism
        model = torch.nn.DataParallel(model)
    params = tools.flatten(model.parameters()) # NOTE: Ordering across runs/nodes seems to be ensured (i.e. only dependent on the model constructor)
    # Finalization
    self._model    = model
    self._name     = name
    self._config   = config
    self._params   = params
    self._gradient = None
    self._defaults = {
      "trainset":  None,
      "testset":   None,
      "loss":      None,
      "criterion": None,
      "optimizer": None }

  def __str__(self):
    """ Compute the "informal", nicely printable string representation of this model.
    Returns:
      Nicely printable string
    """
    return "model %s" % self._name

  @property
  def config(self):
    """ Getter for the immutable configuration.
    Returns:
      Immutable configuration
    """
    return self._config

  def default(self, name, new=None, erase=False):
    """ Get and/or set the named default.
    Args:
      name  Name of the default
      new   Optional new instance, set only if not 'None' or erase is 'True'
      erase Force the replacement by 'None'
    Returns:
      (Old) value of the default
    """
    # Check existence
    if name not in self._defaults:
      raise tools.UnavailableException(self._defaults, name, what="model default")
    # Get current
    old = self._defaults[name]
    # Set if needed
    if erase or new is not None:
      self._defaults[name] = new
    # Return current/old
    return old

  def _resolve_defaults(self, **kwargs):
    """ Resolve the given keyword-arguments with the associated default value.
    Args:
      ... Keyword-arguments, each must have a default if set to None
    Returns:
      In-order given keyword-arguments, with 'None' values replaced with the corresponding default
    """
    res = list()
    for name, value in kwargs.items():
      if value is None:
        value = self.default(name)
        if value is None:
          raise RuntimeError("Missing default %s" % name)
      res.append(value)
    return res

  def run(self, data, training=False):
    """ Run the model at the current parameters for the given input tensor.
    Args:
      data     Input tensor
      training Use training mode instead of testing mode
    Returns:
      Output tensor
    Notes:
      Gradient computation is not enable nor disabled during the run.
    """
    # Set mode
    if training:
      self._model.train()
    else:
      self._model.eval()
    # Compute
    return self._model(data)

  def __call__(self, *args, **kwargs):
    """ Forward call to 'run'.
    Args:
      ... Forwarded (keyword-)arguments
    Returns:
      Forwarded return value
    """
    return self.run(*args, **kwargs)

  def get(self):
    """ Get a reference to the current parameters.
    Returns:
      Flat parameter vector (by reference: future calls to 'set' will modify it)
    """
    return self._params

  def set(self, params, relink=None):
    """ Overwrite the parameters with the given ones.
    Args:
      params Given flat parameter vector
      relink Relink instead of copying (depending on the model, might be faster)
    """
    # Fast path 'set(get())'-like
    if params is self._params:
      return
    # Assignment
    if (self._config.relink if relink is None else relink):
      tools.relink(self._model.parameters(), params)
      self._params = params
    else:
      self._params.copy_(params, non_blocking=self._config["non_blocking"])

  def get_gradient(self):
    """ Get (optionally make each parameter's gradient) a reference to the flat gradient.
    Returns:
      Flat gradient (by reference: future calls to 'set_gradient' will modify it)
    """
    # Fast path
    if self._gradient is not None:
      return self._gradient
    # Flatten (make if necessary)
    gradient = tools.flatten(tools.grads_of(self._model.parameters()))
    self._gradient = gradient
    return gradient

  def set_gradient(self, gradient, relink=None):
    """ Overwrite the gradient with the given one.
    Args:
      gradient Given flat gradient
      relink   Relink instead of copying (depending on the model, might be faster)
    """
    # Fast path 'set(get())'-like
    if gradient is self._gradient:
      return
    # Assignment
    if (self._config.relink if relink is None else relink):
      tools.relink(tools.grads_of(self._model.parameters()), gradient)
      self._gradient = gradient
    else:
      self.get_gradient().copy_(gradient, non_blocking=self._config["non_blocking"])

  def loss(self, dataset=None, loss=None, training=None):
    """ Estimate loss at the current parameters, with a batch of the given dataset.
    Args:
      dataset  Training dataset wrapper to use, use the default one if available
      loss     Loss wrapper to use, use the default one if available
      training Whether this run is for training (instead of testing) purposes, None for guessing (based on whether gradients are computed)
    Returns:
      Loss value
    """
    # Recover the defaults, if missing
    dataset, loss = self._resolve_defaults(trainset=dataset, loss=loss)
    # Sample the train batch
    inputs, targets = dataset.sample(self._config)
    # Guess whether computation is for training, if necessary
    if training is None:
      training = torch.is_grad_enabled()
    # Forward pass
    return loss(self.run(inputs), targets, self._params)

  @torch.enable_grad()
  def backprop(self, dataset=None, loss=None, outloss=False, **kwargs):
    """ Estimate gradient at the current parameters, with a batch of the given dataset.
    Args:
      dataset Training dataset wrapper to use, use the default one if available
      loss    Loss wrapper to use, use the default one if available
      outloss Output the loss value as well
      ...     Additional keyword-arguments forwarded to 'backprop'
    Returns:
      if 'outloss' is True:
        Tuple of:
        · Flat gradient (by reference: future calls to 'backprop' will modify it)
        · Loss value
      else:
        Flat gradient (by reference: future calls to 'backprop' will modify it)
    """
    # Detach and zero the gradient (must be done at each grad to discard computation graph)
    for param in self._params.linked_tensors:
      grad = param.grad
      if grad is not None:
        grad.detach_()
        grad.zero_()
    # Forward and backward passes
    loss = self.loss(dataset=dataset, loss=loss)
    loss.backward(**kwargs)
    # Relink needed if graph of derivatives was created
    # NOTE: It has been observed that each parameters' grad tensor is a new instance in this case; more investigation may be needed to check whether this relink is really necessary, for now this is a safe behavior
    if "create_graph" in kwargs:
      self._gradient = None
    # Return the flat gradient (and the loss if requested)
    if outloss:
      return (self.get_gradient(), loss)
    else:
      return self.get_gradient()

  def update(self, gradient, optimizer=None, relink=None):
    """ Update the parameters using the given gradient, and the given optimizer.
    Args:
      gradient  Flat gradient to apply
      optimizer Optimizer wrapper to use, use the default one if available
      relink    Relink instead of copying (depending on the model, might be faster)
    """
    # Recover the defaults, if missing
    optimizer = self._resolve_defaults(optimizer=optimizer)[0]
    # Set the gradient
    self.set_gradient(gradient, relink=(self._config.relink if relink is None else relink))
    # Perform the update step
    optimizer.step()

  @torch.no_grad()
  def eval(self, dataset=None, criterion=None):
    """ Evaluate the model at the current parameters, with a batch of the given dataset.
    Args:
      dataset   Testing dataset wrapper to use, use the default one if available
      criterion Criterion wrapper to use, use the default one if available
    Returns:
      Arithmetic mean of the criterion over the next minibatch
    """
    # Recover the defaults, if missing
    dataset, criterion = self._resolve_defaults(testset=dataset, criterion=criterion)
    # Sample the test batch
    inputs, targets = dataset.sample(self._config)
    # Compute and return the evaluation result
    return criterion(self.run(inputs), targets)

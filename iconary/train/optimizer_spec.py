import logging
import re
from typing import List, Tuple

import torch
import transformers
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import AdamW, Adafactor
from allennlp.common import Registrable, FromParams


class LearnScheduleSpec(Registrable):
  """Builds a transfomer learning rate schedule to use"""
  def build_learn_schedule(self, optimizer, total_steps):
    raise NotImplementedError()


class OptimizerSpec(Registrable):
  """Builds an optimizer to train on"""
  def build_optimizer(self, model, total_steps):
    raise NotImplementedError()


@LearnScheduleSpec.register("triangle")
class Triangle(LearnScheduleSpec):

  def __init__(self, warmup):
    self.warmup = warmup

  def build_learn_schedule(self, optimizer, total_steps):
    if self.warmup is None:
      return None
    return transformers.optimization.get_linear_schedule_with_warmup(
      optimizer, self.warmup, total_steps)


class ParameterGroup(FromParams):
  def __init__(self, group_name, regex, overrides, allow_empty=False):
    self.group_name = group_name
    self.regex = regex
    self.overrides = overrides
    self.allow_empty = allow_empty


@LearnScheduleSpec.register("linear-warmup")
class LinearWarmupSpec(LearnScheduleSpec):
  def __init__(self, warmup):
    self.warmup = warmup

  def build_learn_schedule(self, optimizer, total_steps):

    def lr_fn(current_step):
      current_step += 1
      if current_step < self.warmup:
        return float(current_step) / float(self.warmup)
      return 1.0
    return LambdaLR(optimizer, lr_fn, -1)


class LearningRateSchedule(Registrable):
  def get_learning_factor(self, step):
    raise NotImplementedError()


@OptimizerSpec.register("adam-w")
class AdamWSpec(OptimizerSpec):

  def __init__(self, lr, weight_decay=0.0, betas: Tuple[float, float] = (0.9, 0.999),
               correct_bias=True, parameter_groups: List[ParameterGroup]=None):
    self.lr = lr
    self.correct_bias = correct_bias
    self.betas = betas
    self.weight_decay = weight_decay
    self.parameter_groups = parameter_groups

  def build_optimizer(self, model: torch.nn.Module, total_steps):
    if self.parameter_groups is None:
      parameters = model.parameters()
    else:
      name_to_group = {}
      group_to_params = {}
      for group in self.parameter_groups:
        if group.group_name in name_to_group:
          raise ValueError()
        name_to_group[group.group_name] = group
        group_to_params[group.group_name] = []
      default = []
      for name, param in model.named_parameters():
        any_match = False
        for group in self.parameter_groups:
          if re.match(group.regex, name):
            group_to_params[group.group_name].append(param)
            any_match = True
            break
        if not any_match:
          default.append(param)

      parameters = []
      for group_name, params in group_to_params.items():
        group = name_to_group[group_name]
        if len(params) == 0 and not group.allow_empty:
          raise RuntimeError(f"Group {group.group_name} empty")
        if len(params) > 0:
          logging.info(f"Found {len(params)} in parameter group {group_name}")
          param_group = dict(group.overrides)
          param_group["params"] = params
          parameters.append(param_group)
      if len(default) > 0:
        logging.info(f"Found {len(default)} in remaining parameters in default group")
        if "default" in parameters:
          raise ValueError()
        default_params = dict(params=default)
        parameters.append(default_params)

    return AdamW(
      parameters, lr=self.lr, correct_bias=self.correct_bias,
      weight_decay=self.weight_decay, betas=self.betas)


@OptimizerSpec.register("adafactor")
class AdaFactorSpec(OptimizerSpec):

  def __init__(self, lr, relative_step, warmup_init, scale_parameter,
               beta1=None, max_grad_norm=None, epsilon=1e-30,
               parameter_groups: List[ParameterGroup]=None):
    self.lr = lr
    self.epsilon = epsilon
    self.beta1 = beta1
    self.relative_step = relative_step
    self.warmup_init = warmup_init
    self.max_grad_norm = max_grad_norm
    self.parameter_groups = parameter_groups
    self.scale_parameter = scale_parameter

  def build_optimizer(self, model: torch.nn.Module, total_steps):
    if self.parameter_groups is None:
      parameters = model.parameters()
    else:
      name_to_group = {}
      group_to_params = {}
      for group in self.parameter_groups:
        if group.group_name in name_to_group:
          raise ValueError()
        name_to_group[group.group_name] = group
        group_to_params[group.group_name] = []
      default = []
      for name, param in model.named_parameters():
        any_match = False
        for group in self.parameter_groups:
          if re.match(group.regex, name):
            group_to_params[group.group_name].append(param)
            any_match = True
            break
        if not any_match:
          default.append(param)

      parameters = []
      for group_name, params in group_to_params.items():
        group = name_to_group[group_name]
        if len(params) == 0 and not group.allow_empty:
          raise RuntimeError(f"Group {group.group_name} empty")
        if len(params) > 0:
          logging.info(f"Found {len(params)} in parameter group {group_name}")
          param_group = dict(group.overrides)
          param_group["params"] = params
          parameters.append(param_group)
      if len(default) > 0:
        logging.info(f"Found {len(default)} in remaining parameters in default group")
        if "default" in parameters:
          raise ValueError()
        default_params = dict(params=default)
        parameters.append(default_params)

    opt = Adafactor(parameters, self.lr,
                    beta1=self.beta1,
                    eps=(self.epsilon, 1e-3),
                    scale_parameter=self.scale_parameter,
                    warmup_init=self.warmup_init, relative_step=self.relative_step)
    if hasattr(model, "set_optimizer"):  # TODO move to trainer?
      model.set_optimizer(opt)
    return opt

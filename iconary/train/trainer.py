import json
import logging
import os
import socket
from datetime import datetime
from os import mkdir, listdir, makedirs
from os.path import join, exists
from shutil import rmtree
from typing import Union, Dict

import torch
from allennlp.common import Params
from allennlp.common.from_params import FromParams
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset, Dataset

from iconary.data.datasets import IconaryDataset
from iconary.models.iconary_model import IconaryModel
from iconary.train.dataset_iterators import ImsituGameIterator
from iconary.train.evaluator import Evaluator, ClfEvaluator
from iconary.train.optimizer_spec import OptimizerSpec, LearnScheduleSpec
from iconary.utils import utils
from iconary.utils.utils import to_device, dump_json_object
from iconary.utils.to_params import to_params

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
  SummaryWriter = None

from tqdm import tqdm


def select_subdir(output_dir, target=None):
  prefix = "" if target is None else target + "-"
  i = 0
  while True:
    candidate = join(output_dir, prefix + "r" + str(i))
    if not exists(candidate):
      try:
        mkdir(candidate)
        return candidate
      except FileExistsError:
        pass
    i += 1


def clear_if_nonempty(output_dir):
  if output_dir:
    if exists(output_dir) and listdir(output_dir):
      if input("%s is non-empty, override (y/n)?" % output_dir).strip() == "y":
        rmtree(output_dir)
      else:
        raise ValueError(
          "Output directory ({}) already exists and is not empty.".format(output_dir))


class ModelTrainer(FromParams):

  @classmethod
  def from_params(
        cls,
        params: Params,
        constructor_to_call=None,
        constructor_to_inspect=None,
        **extras,
    ):
    if params["evaluator"] is None:
      params["evaluator"] = ClfEvaluator()
    return super().from_params(params, constructor_to_call, constructor_to_inspect, **extras)

  def __init__(
    self,
    train_dataset: IconaryDataset,
    validation_dataset: IconaryDataset,
    train_batch_size, n_epochs,
    optimizer_builder: OptimizerSpec,
    init_from: str=None,

    # Optional optimization parameters
    train_iterator: ImsituGameIterator=None,
    learning_schedule_builder: LearnScheduleSpec=None,
    gradient_accumulation_steps=1,
    max_grad_norm=None,
    num_workers: int=0,

    # Evaluation parameters
    evaluator: Evaluator=None,
    eval_iterator: ImsituGameIterator=None,
    valid_batch_size=None,
    symlink_last_state_if_possible=True,
    eval_each_epoch=True,
    eval_n_steps=None,

    save_best_model=True,
    save_each_epoch=True,
    best_model_key="acc",

    # Logging parameters
    loss_logging_ema=0.995,
    tb_log=True,
    tb_log_intervals=10,
    new_line_after_eval=True
  ):
    self.save_best_model = save_best_model
    self.init_from = init_from
    self.tb_log_intervals = tb_log_intervals
    self.tb_log = tb_log
    self.save_each_epoch = save_each_epoch
    if save_each_epoch:
      self.save_best_model = False
    self.symlink_last_state_if_possible = symlink_last_state_if_possible
    self.max_grad_norm = max_grad_norm
    self.train_iterator = train_iterator
    self.eval_iterator = eval_iterator
    self.eval_each_epoch = eval_each_epoch
    self.train_dataset = train_dataset
    self.learning_schedule_builder = learning_schedule_builder
    self.validation_dataset = validation_dataset
    self.train_batch_size = train_batch_size
    self.evaluator = evaluator
    self.eval_n_steps = eval_n_steps
    if valid_batch_size is None:
      valid_batch_size = train_batch_size // gradient_accumulation_steps
    self.valid_batch_size = valid_batch_size
    self.gradient_accumulation_steps = gradient_accumulation_steps
    self.n_epochs = n_epochs
    self.optimizer_builder = optimizer_builder
    self.new_line_after_eval = new_line_after_eval
    self.loss_logging_ema = loss_logging_ema
    self.best_model_key = best_model_key
    self.num_workers = num_workers

  def _eval(self, model, valid_loader, cuda, device, summary_writer,
            n_steps, best_model_file, best_valid_score):
    model.eval()

    for batch in tqdm(valid_loader, desc="valid", ncols=100):
      if cuda:
        batch = to_device(batch, device)
      with torch.no_grad():
        output = model(**batch)
      self.evaluator.evaluate_batch(output)

    model.train()
    if self.new_line_after_eval:
      print()

    results = self.evaluator.get_stats()
    self.evaluator.clear()
    logging.info(", ".join(f"{k}={v:0.3f}" for k, v in results.items()))
    if summary_writer is not None:
      for result_name, val in results.items():
        summary_writer.add_scalar(f"valid/{result_name}", val, n_steps)

    saved_model = False
    if best_model_file is not None and self.save_best_model:
      valid_token_acc = results[self.best_model_key]
      if best_valid_score is None or best_valid_score < valid_token_acc:
        saved_model = True
        best_valid_score = valid_token_acc
        logging.info(f"Saving as best model")
        torch.save(model.state_dict(), best_model_file)
    return best_valid_score, saved_model

  def train(self, model: IconaryModel, output_dir, cuda=None):
    bs = self.train_batch_size
    if self.gradient_accumulation_steps != 1:
      if bs % self.gradient_accumulation_steps != 0:
        raise ValueError()
      bs = self.train_batch_size // self.gradient_accumulation_steps
      logging.info(f"Do {self.gradient_accumulation_steps} grad accumlation steps "
                   f"with batch size {bs}")

    logging.info("Initializing")
    model.initialize()
    train = self.train_dataset.load()

    if self.train_iterator is not None:
      train = self.train_iterator.build_dataset(train)

    if self.train_iterator.is_batch_iterator():
      def _collate(x):
        return model.collate_train(x[0])
      if isinstance(train, IterableDataset):
        train_loader = DataLoader(
          train, batch_size=1, collate_fn=_collate, shuffle=False, num_workers=self.num_workers)
      else:
        train_loader = DataLoader(
          train, batch_size=1, collate_fn=_collate, shuffle=True, num_workers=self.num_workers)
    else:
      train_loader = DataLoader(
        train, batch_size=bs, collate_fn=model.collate_train, shuffle=True, num_workers=self.num_workers)

    if self.validation_dataset is not None:
      validation = self.validation_dataset.load()
      if self.eval_iterator is not None:
        validation = self.eval_iterator.build_dataset(validation)
      valid_loader = DataLoader(
        validation, batch_size=self.valid_batch_size, collate_fn=model.collate,
        shuffle=True, num_workers=self.num_workers)
    else:
      valid_loader = None

    if self.init_from is not None:
      logging.info(f"Loading weights from {self.init_from}")
      model.load_state_dict(torch.load(self.init_from))

    if cuda is None:
      cuda = torch.cuda.is_available()
      if cuda:
        logging.info("cuda available, training on cuda")
      else:
        logging.info("cuda not available, training on cpu")
    elif cuda:
      if torch.cuda.is_available():
        raise ValueError()

    if cuda:
      device = torch.device("cuda")
      model = model.to(device)
    else:
      device = torch.device("cpu")

    total_steps = (len(train_loader) * self.n_epochs) // self.gradient_accumulation_steps
    optimizer = self.optimizer_builder.build_optimizer(model, total_steps)
    if self.learning_schedule_builder is not None:
      learning_schedule = self.learning_schedule_builder.build_learn_schedule(
        optimizer, total_steps)
    else:
      learning_schedule = None

    summary_writer = None
    if output_dir:
      subdir = select_subdir(output_dir)
      logging.info("Saving run to %s" % subdir)

      model_output = subdir

      if self.tb_log:
        summary_writer = SummaryWriter(join(subdir, "log"))

      with open(join(model_output, "runtime.json"), "w") as f:
        json.dump(dict(
          hostname=socket.gethostname(), date=datetime.now().strftime("%m%d-%H%M%S"),
          n_gpu=(1 if cuda else 0),
        ), f, indent=2)

      dump_json_object(dict(done=False), join(model_output, "status.json"))
      output_model_file = join(model_output, "state.pth")
      best_model_file = join(subdir, "best-state.pth")
    else:
      subdir = None
      best_model_file = None
      output_model_file = None

    best_valid_score = None

    last_save = None
    loss_ema = 0
    n_steps = 0
    n_gradients_accumulated = 0
    accumulated_loss = 0
    loss_decay = self.loss_logging_ema
    optimizer.zero_grad()

    for epoch in range(self.n_epochs):
      model.train()

      pbar = tqdm(train_loader, desc="train", ncols=100, disable=False)
      for i, batch in enumerate(pbar):
        if cuda:
          batch = to_device(batch, device)

        out = model(**batch)
        loss = out[0]

        if self.gradient_accumulation_steps > 1:
          loss /= self.gradient_accumulation_steps

        loss.backward()

        if self.gradient_accumulation_steps > 1:
          # Note loss is already divided by `self.gradient_accumulation_steps`
          accumulated_loss += loss.item()
          n_gradients_accumulated += 1
          if n_gradients_accumulated == self.gradient_accumulation_steps:
            loss = accumulated_loss
            n_gradients_accumulated = 0
            accumulated_loss = 0
          else:
            continue
        else:
          if not isinstance(loss, float):
            loss = loss.item()

        if self.max_grad_norm:
          torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm, norm_type=2)
        optimizer.step()

        if learning_schedule is not None:
          learning_schedule.step()
        optimizer.zero_grad()

        n_steps += 1
        loss_ema = loss_ema * loss_decay + loss * (1 - loss_decay)
        corrected_loss_ema = (loss_ema / (1 - loss_decay ** n_steps))
        pbar.set_description("loss=%.4f" % corrected_loss_ema, refresh=False)

        if summary_writer is not None and n_steps % self.tb_log_intervals == 0:
          summary_writer.add_scalar("train/loss-smoothed", corrected_loss_ema, n_steps)
          summary_writer.add_scalar("train/loss", loss, n_steps)

        if (valid_loader is not None and
            self.eval_n_steps and n_steps % self.eval_n_steps == 0):
          best_valid_score, is_saved = self._eval(
            model, valid_loader, cuda, device, summary_writer, n_steps,
            best_model_file, best_valid_score)
          if is_saved:
            last_save = n_steps

      if self.save_each_epoch and subdir is not None:
        torch.save(model.state_dict(), join(subdir, f"state-ep{epoch+1}.pth"))

      is_last_epoch = epoch == self.n_epochs - 1
      if valid_loader is not None and (
          is_last_epoch or
          self.eval_each_epoch is True or
          isinstance(self.eval_each_epoch, int) and epoch % self.eval_each_epoch == 0
      ):
        best_valid_score, is_saved = self._eval(
          model, valid_loader, cuda, device, summary_writer, n_steps,
          best_model_file, best_valid_score)
        if is_saved:
          last_save = n_steps

    if output_dir is not None:
      if not self.save_each_epoch:
        if last_save == n_steps and self.symlink_last_state_if_possible:
          os.symlink(best_model_file, output_model_file)
        else:
          torch.save(model.state_dict(), output_model_file)
      dump_json_object(dict(done=True), join(subdir, "status.json"))
    return subdir


def train_another_model(output_dir, cuda=None):
  utils.import_all()
  model_file = join(output_dir, "model.json")
  trainer_file = join(output_dir, "trainer.json")
  trainer = ModelTrainer.from_params(Params.from_file(trainer_file))
  model = IconaryModel.from_params(Params.from_file(model_file))
  trainer.train(model, output_dir, cuda)


def init_model(
    trainer: Union[ModelTrainer, Params],
    model: Union[IconaryModel, Params],
    output_dir
):
  clear_if_nonempty(output_dir)
  makedirs(output_dir, exist_ok=True)
  if isinstance(trainer, Params):
    trainer.to_file(join(output_dir, "trainer.json"))
  else:
    Params(to_params(trainer, ModelTrainer)).to_file(join(output_dir, "trainer.json"))
  if isinstance(model, Params):
    model.to_file(join(output_dir, "model.json"))
  else:
    Params(to_params(model, IconaryModel)).to_file(join(output_dir, "model.json"))


def train_model(
    trainer: Union[ModelTrainer, Params],
    model: Union[IconaryModel, Params],
    output_dir, cuda=None
):
  if output_dir:
    # Make sure we have a place to save
    clear_if_nonempty(output_dir)
    makedirs(output_dir, exist_ok=True)
    if isinstance(trainer, Params):
      trainer.to_file(join(output_dir, "trainer.json"))
      trainer = ModelTrainer.from_params(trainer)
    else:
      Params(to_params(trainer, None)).to_file(join(output_dir, "trainer.json"))

    if isinstance(model, Params):
      model.to_file(join(output_dir, "model.json"))
      model = IconaryModel.from_params(model)
    else:
      Params(to_params(model, IconaryModel)).to_file(join(output_dir, "model.json"))

  else:
    if isinstance(model, Params):
      model = IconaryModel.from_params(model)
    if isinstance(trainer, Params):
      trainer = ModelTrainer.from_params(trainer)

  return trainer.train(model, output_dir, cuda)

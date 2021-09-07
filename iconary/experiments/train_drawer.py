import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from iconary.data.datasets import IconaryDataset
from iconary.models.drawing_to_text import DrawingEncoder
from iconary.models.game_phrase_to_text import GamePhraseMarkKnown
from iconary.models.t5_models import TDrawer
from iconary.train.dataset_iterators import AllDrawingsPreviousGuess
from iconary.train.evaluator import ClfEvaluator
from iconary.train.optimizer_spec import AdaFactorSpec
from iconary.train.trainer import ModelTrainer, train_model
from iconary.utils.utils import add_stdout_logger


def main():
  parser = argparse.ArgumentParser(description="Train TDrawer")

  # Model hyperparameters, we use the defaults for TGuesser
  parser.add_argument("--pretrained_model",
                      choices=["t5-small", "t5-base", "t5-large", "t5-3b"],
                      default="t5-base")
  parser.add_argument("--no_number_init", action="store_true")
  parser.add_argument("--no_icon_init", action="store_true")
  parser.add_argument("--mark_known", action="store_true")
  parser.add_argument("--freeze_embed", action="store_true")
  parser.add_argument("--no_train_constraints", action="store_true")
  parser.add_argument("--sort", default=None)

  # Optimization hyperparameters
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--epochs", type=int, default=2)
  parser.add_argument("--batch_size", type=int, default=32)

  # Training parameters
  parser.add_argument("--output_dir")
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--grad_accumulation", type=int, default=1)
  parser.add_argument("--save_each_epoch", action="store_true")

  args = parser.parse_args()

  add_stdout_logger()
  dbg = args.debug

  optimizer = AdaFactorSpec(
    lr=args.lr, warmup_init=False, relative_step=False, scale_parameter=False)
  lr_schedule = None

  if dbg:
    tr = IconaryDataset("train", 200)
    valid = IconaryDataset("ind-valid", 50)
  else:
    tr = IconaryDataset("train")
    valid = IconaryDataset("ind-valid")

  trainer_params = ModelTrainer(
    train_dataset=tr,
    train_iterator=AllDrawingsPreviousGuess(),
    validation_dataset=valid,
    eval_iterator=AllDrawingsPreviousGuess(),
    train_batch_size=args.batch_size,
    n_epochs=args.epochs,
    max_grad_norm=None,
    tb_log_intervals=5,
    gradient_accumulation_steps=args.grad_accumulation,
    optimizer_builder=optimizer,
    learning_schedule_builder=lr_schedule,
    evaluator=ClfEvaluator(),
    tb_log=True,
    save_best_model=False,
    save_each_epoch=args.save_each_epoch,
    num_workers=0 if dbg else 3
  )

  model_params = TDrawer(
    pretrained_model="t5-small" if args.debug else args.pretrained_model,
    game_phrase_encoder=GamePhraseMarkKnown(mark_known=True, mode="star"),
    drawing_encoder=DrawingEncoder(
      canvas_width=800.0,
      canvas_height=450.0,
      num_x_location_bins=32,
      num_y_location_bins=18,
      xy_top_left=False,
      icon_size=80,
      num_rotation_buckets=8,
      scale_bucket_boundaries=[0, 0.25, 0.5, 0.75, 0.9, 1.1, 2, 4, 6, 8, 10],
      icon_dictionary=True,
      max_icons=29,
      clipping=3,
      sort=args.sort,
    ),
    train_with_constraints=not args.no_train_constraints,
    icon_name_init=not args.no_icon_init,
    number_init=None if args.no_number_init else "count",
    freeze_word_embed=args.freeze_embed
  )

  train_model(trainer_params, model_params, args.output_dir)


if __name__ == '__main__':
  main()

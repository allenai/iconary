import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from iconary.data.datasets import IconaryDataset
from iconary.models.constraints_to_text import KnownWords
from iconary.models.drawing_to_text import IconCounts, IconNames, IconCountsWithModifiers
from iconary.models.game_to_text import FillInPhraseBlanks, GuessPhrase
from iconary.models.t5_models import TGuesser
from iconary.train.dataset_iterators import AllDrawingsRandomGuess
from iconary.train.evaluator import ClfEvaluator
from iconary.train.optimizer_spec import AdaFactorSpec
from iconary.train.trainer import train_model, ModelTrainer
from iconary.utils.utils import add_stdout_logger


def main():
  parser = argparse.ArgumentParser(description="Train TGuesser")

  # Model hyperparameters
  parser.add_argument("--pretrained_model",
                      choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
                      default="t5-base")
  parser.add_argument(
    "--encoding", default="colon", help="Why to encode the previous guesses/constraints")
  parser.add_argument(
    "--drawing", default="modifiers-xorder", help="Why to encode the drawing")
  parser.add_argument("--no_freeze_embed", action="store_true",
                      help="Don't freeze the embeddings (usually harms performance on the OOD dataset)")

  # Optimization hyperparameters
  parser.add_argument("--lr", type=float, default=5e-5)
  parser.add_argument("--epochs", type=int, default=1)
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--max_grad_norm", type=float, default=None)

  # Training parameters
  parser.add_argument("--grad_accumulation", type=int, default=1)
  parser.add_argument("--save_each_epoch", action="store_true",
                      help='Save the model after each epoch as well at the end of training')
  parser.add_argument("--debug", action="store_true",
                      help="Run on a small sample if the data")
  parser.add_argument("--output_dir", help="Where to save the model")

  args = parser.parse_args()

  add_stdout_logger()
  dbg = args.debug

  optimizer = AdaFactorSpec(
    lr=args.lr, warmup_init=False, relative_step=False, scale_parameter=False)
  if dbg:
    tr = IconaryDataset("train", 200)
    valid = IconaryDataset("ind-valid", 50)
  else:
    tr = IconaryDataset("train")
    valid = IconaryDataset("ind-valid")

  if args.drawing == "counts":
    drawing_encoder = IconCounts(comma=True, num_first=True)
  elif args.drawing == "names":
    drawing_encoder = IconNames()
  elif args.drawing == "modifiers-xorder":
    drawing_encoder = IconCountsWithModifiers(order="x", encoding_version=3)
  else:
    raise NotImplementedError(args.drawing)

  if args.encoding == "colon":
    game_to_text = FillInPhraseBlanks(
      drawing_prefix=["drawing:"],
      drawing_encoder=drawing_encoder,
      phrase_prefix=["phrase:"],
    )
  elif args.encoding == "none":
    game_to_text = FillInPhraseBlanks(
      drawing_prefix=[],
      drawing_encoder=drawing_encoder,
      phrase_prefix=[],
    )
  elif args.encoding == "phrase":
    game_to_text = FillInPhraseBlanks(
      drawing_prefix=["There", "is"],
      drawing_encoder=drawing_encoder,
      phrase_prefix=["I", "saw"],
    )
  elif args.encoding == "guess_phrase_colon":
    game_to_text = GuessPhrase(
      constraint_encoder=KnownWords(["phrase:"], blank_token="_"),
      drawing_prefix=["drawing:"],
      drawing_encoder=drawing_encoder,
      constraints_first=False
    )
  else:
    raise NotImplementedError(args.encoding)

  model_params = TGuesser(
    pretrained_model="t5-small" if dbg else args.pretrained_model,
    game_to_text=game_to_text,
    freeze_embed=not args.no_freeze_embed,
  )

  trainer_params = ModelTrainer(
    train_dataset=tr,
    train_iterator=AllDrawingsRandomGuess(),
    validation_dataset=valid,
    eval_iterator=AllDrawingsRandomGuess(),
    evaluator=ClfEvaluator(),
    train_batch_size=args.batch_size,
    n_epochs=args.epochs,
    max_grad_norm=args.max_grad_norm,
    tb_log_intervals=5,
    gradient_accumulation_steps=args.grad_accumulation,
    optimizer_builder=optimizer,
    learning_schedule_builder=None,
    tb_log=True,
    save_best_model=False,
    save_each_epoch=args.save_each_epoch,
    num_workers=0 if dbg else 3
  )

  train_model(trainer_params, model_params, args.output_dir)


if __name__ == '__main__':
  main()
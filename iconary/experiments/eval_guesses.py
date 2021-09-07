import argparse
import json
import logging

from allennlp.common import Params

from iconary.data.datasets import IconaryDataset, Dataset
from iconary.eval.eval_guess import eval_guesses
from iconary.models.generation import load_guesses, dictionary_to_guesses
from iconary.utils import utils
from iconary.utils.utils import load_json_object


def main():
  parser = argparse.ArgumentParser(
    description="Compute automatic metrics on guesses for human/human games")
  parser.add_argument("eval_file", help="File produced by `generate_guesses.py`")

  args = parser.parse_args()
  utils.add_stdout_logger()

  logging.info(f"Loading file {args.eval_file}")
  data = load_json_object(args.eval_file)

  # backwards compatibility hacks
  if data["dataset"] == "iconary-ind-dev-unfiltered-v2":
    dataset = IconaryDataset("ind-valid")
  elif data["dataset"] == "ood-dev-v2":
    dataset = IconaryDataset("ood-valid")
  else:
    dataset = IconaryDataset.from_params(Params(data["dataset"]))

  logging.info(f"Evaluation was on {dataset.get_name()}, evaluating on that dataset")
  guesses = dictionary_to_guesses(data["guesses"])

  logging.info(f"Starting evaluation")
  results = eval_guesses(dataset.load(), guesses, dataset.is_ood)
  print(json.dumps(results))


if __name__ == '__main__':
  main()
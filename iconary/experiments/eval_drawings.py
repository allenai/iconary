import argparse
import json
import logging

from allennlp.common import Params

from iconary.data.datasets import IconaryDataset
from iconary.eval.eval_drawing import eval_drawings
from iconary.experiments.generate_drawings import load_drawing_predictions_from_json
from iconary.utils import utils
from iconary.utils.utils import load_json_object


def main():
  parser = argparse.ArgumentParser(
    description="Compute automatic metrics on drawings for human/human games")
  parser.add_argument("eval_file", help="file from `generate_drawings.py`")

  args = parser.parse_args()
  utils.add_stdout_logger()

  logging.info(f"Loading file {args.eval_file}")
  data = load_json_object(args.eval_file)

  dataset: IconaryDataset = IconaryDataset.from_params(Params(data["dataset"]))

  logging.info(f"Evaluation was on {dataset.get_name()}, evaluating on that dataset")
  drawings = load_drawing_predictions_from_json(data["drawings"])

  logging.info(f"Starting evaluation")
  games = dataset.load()
  if dataset.sample is None:
    dataset.sample = None
    ref = dataset.load()
  else:
    ref = games

  results = eval_drawings(games, ref, drawings)
  print(json.dumps(results))


if __name__ == '__main__':
  main()
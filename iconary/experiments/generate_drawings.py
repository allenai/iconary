import argparse
import json
import logging
from dataclasses import dataclass, replace, asdict
from datetime import datetime
from queue import Empty
from typing import Dict

import torch
from allennlp.common import FromParams
from tqdm import tqdm

from iconary.data.datasets import IconaryDataset
from iconary.eval.eval_drawing import eval_drawings, DrawingPredictions
from iconary.models.generation import BeamSearchConfig, AllenNLPBeamSearcher, generate_drawings, \
  GeneratedDrawing
from iconary.utils import utils

from iconary.utils.load_model import load_model
from iconary.utils.utils import dump_json_object
from iconary.utils.to_params import to_params


@dataclass
class DrawingConfig(FromParams):

  """Number of drawings to save"""
  top_n: int

  """Beam search setup to use"""
  beam_search: BeamSearchConfig


def save_drawings(output_file: str, prediction: DrawingPredictions,
                  dataset: IconaryDataset, evaluation: Dict[str, float], config: DrawingConfig):
  """Saves drawings and meta-data"""

  out = dict(
    dataset_name=dataset.get_name(),
    dataset=to_params(dataset, IconaryDataset),
    evaluation=evaluation,
    config=None if config is None else to_params(config, DrawingConfig),
    date=datetime.now().strftime("%m%d-%H%M%S"),
    drawings=asdict(prediction)
  )
  dump_json_object(out, output_file)


def load_drawing_predictions_from_json(obj: Dict) -> DrawingPredictions:
  phrase_to_drawings = {k: [GeneratedDrawing.from_dict(p) for p in v] for
                        k, v in obj["phrase_to_drawings"].items()}
  game_id_to_drawings = {}
  for game_id, pre in obj["game_id_to_drawings"].items():
    game_id_to_drawings[game_id] = \
      [[GeneratedDrawing.from_dict(x) for x in lst] for lst in  pre]
  return DrawingPredictions(
    phrase_to_drawings,
    game_id_to_drawings,
    obj["game_id_to_initial_phrase"]
  )


def generate_drawings_from_queue(rank, devices, model, queue, bs, top_n, out_q, batch_size=8):
  model = model.to('cuda:' + str(device_id))
  done = False
  while not done:
    tasks = []
    games = []
    for _ in range(batch_size):
      try:
        task_id, game = queue.get(block=False)
        tasks.append(task_id)
        games.append(game)
      except Empty:
        done = True
        break
    if len(tasks) > 0:
      outs = [x[:top_n] for x in generate_drawings(games, model, bs)]
      for t, o in zip(tasks, outs):
        out_q.put((t, o))


def generate_from_models(model, ds, ref_ds, config: DrawingConfig,
                         output_file, batch_size=8, epoch=None):
  """Generate drawings on states in human/human games

  :param model: Model directory to run from
  :param ds: IconaryDataset to generate from
  :param ref_ds: IconaryDataset to use as references during evaluations
  :param config: Generation configuration
  :param output_file: Where to save the output
  :param batch_size: Batch size of generation beam search
  :param epoch: Specify which epoch checkpoint to load if not the latest
  :return: The evaluation results
  """
  if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()
  else:
    device = torch.device("cpu")
    n_gpus = 0

  games = ds.load()
  if ref_ds is not None:
    refs = ref_ds.load()
  else:
    refs = games

  if output_file is not None:
    logging.info(f"Will save output into {output_file}")

  model = load_model(model, epoch=epoch)
  bs = config.beam_search.get_beam_searcher(model)
  game_id_to_initial_phrase = {}
  seen_phrases = set()

  # Compute all the drawings we will need to generate
  drawing_tasks = []
  for game in games:
    phrase = " ".join(game.game_phrase)
    game_id_to_initial_phrase[game.id] = phrase

    if phrase not in seen_phrases:
      drawing_tasks.append((phrase, replace(game, game_states=[])))
      seen_phrases.add(phrase)

    for i in range(1, len(game.game_states)):
      drawing_tasks.append(((game.id, i-1), replace(game, game_states=game.game_states[:i])))

  assert len(set(x[0] for x in drawing_tasks)) == len(drawing_tasks)

  # Get the drawing
  top_n = config.top_n
  task_to_drawing = {}
  if n_gpus <= 1:
    model = model.to(device)
    batched_tasks = [drawing_tasks[i:i+batch_size] for i in range(0, len(drawing_tasks), batch_size)]
    for batch in tqdm(batched_tasks, ncols=100, desc="generate", disable=False):
      task_list = [x[0] for x in batch]
      game_list = [x[1] for x in batch]
      out = generate_drawings(game_list, model, bs)
      for t, o in zip(task_list, out):
        task_to_drawing[t] = o[:top_n]

  else:
    # To use multiprocessing with CUDA we need to use a 'spawn' context
    ctx = torch.multiprocessing.get_context("spawn")

    logging.info(f"Splitting up generation on {n_gpus} devices")
    q = ctx.Queue()
    for x in drawing_tasks:
      q.put(x)
    out_q = ctx.Queue()

    args = (list(range(n_gpus)), model, q, bs, top_n, out_q, batch_size)
    context = torch.multiprocessing.spawn(
      generate_drawings_from_queue, nprocs=n_gpus, args=args, join=False)

    pbar = tqdm(total=len(drawing_tasks), ncols=100, desc="generate", disable=False)
    while pbar.total != pbar.n:
      task_id, drawing = out_q.get()
      task_to_drawing[task_id] = drawing
      pbar.update(1)

    while not context.join():
      pass

  # Re-organize the results as dictionaries
  phrase_to_drawings = {}
  game_id_to_drawings = {x.id: [None]*(len(x.game_states)-1) for x in games}
  for task_id, drawing in task_to_drawing.items():
    if isinstance(task_id, str):
      phrase_to_drawings[task_id] = drawing
    else:
      game_id_to_drawings[task_id[0]][task_id[1]] = drawing
  predictions = DrawingPredictions(
    game_id_to_drawings=game_id_to_drawings,
    game_id_to_initial_phrase=game_id_to_initial_phrase,
    phrase_to_drawings=phrase_to_drawings
  )

  logging.info("Evaluating...")
  results = eval_drawings(games, refs, predictions)
  logging.info(json.dumps(results, indent=2))

  if output_file:
    logging.info(f"Saving output into {output_file}")
    save_drawings(output_file, predictions, ds, results, config)

  return results


def main():
  parser = argparse.ArgumentParser(
    description="Generate drawings the model would have made for game states in human/human games")
  parser.add_argument("model")
  parser.add_argument("--output_file", help="Where to save the drawing")
  parser.add_argument("--dataset", choices=IconaryDataset.SPLITS, default=["ind-valid"], nargs="+",
                      help="Which dataset(s) to run on")
  parser.add_argument("--beam_size", type=int, default=20)
  parser.add_argument("--max_steps", type=int, default=120)
  parser.add_argument("--epoch", type=int, default=None)
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--top_n", type=int, default=None)
  parser.add_argument("--sample", type=int, default=None, help="Subsample the test data")

  args = parser.parse_args()
  utils.add_stdout_logger()

  config = DrawingConfig(
    top_n=args.top_n,
    beam_search=AllenNLPBeamSearcher(args.beam_size, None, args.max_steps)
  )

  model = utils.select_run_dir(args.model)

  logging.info("Generating from config: " + str(config))

  for dataset in args.dataset:
    ds = IconaryDataset(dataset, sample=args.sample)
    ref = IconaryDataset(dataset)
    logging.info(f"Evaluating on dataset {ds.get_name()}")
    generate_from_models(model, ds, ref, config, args.output_file,
                         batch_size=args.batch_size, epoch=args.epoch)


if __name__ == '__main__':
  main()
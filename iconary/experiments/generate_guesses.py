import argparse

import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from os import listdir, makedirs
from os.path import join, dirname, exists
from queue import Empty
from typing import Dict

import numpy as np
import torch
from allennlp.common import FromParams, Params
from tqdm import tqdm

from iconary.data.datasets import IconaryGame, Dataset, IconaryDataset
from iconary.eval.eval_guess import eval_guesses
from iconary.models.generation import BeamSearchConfig, guesses_to_dictionary, \
  dictionary_to_guesses, generate_guess_sequence, generate_guess_batch, AllenNLPBeamSearcher
from iconary.utils import utils
from iconary.utils.load_model import load_model
from iconary.utils.utils import dump_json_object, load_json_object, is_model_dir
from iconary.utils.to_params import to_params


@dataclass
class GuessingConfig(FromParams):
  top_n_to_save: int
  n_guesses_per_state: int
  beam_search: BeamSearchConfig


def save_guesses(config, dataset: IconaryDataset, guess, output_file,
                 evaluation: Dict[str, float]):
  out = dict(
    dataset_name=dataset.get_name(),
    dataset=to_params(dataset, IconaryDataset),
    evaluation=evaluation,
    config=None if config is None else to_params(config, GuessingConfig),
    date=datetime.now().strftime("%m%d-%H%M%S"),
    guesses=guesses_to_dictionary(guess)
  )
  dump_json_object(out, output_file)


def generate_guess_sequence_from_all_states(
    game: IconaryGame, model, beam_searcher, n_iterations, n_to_return=1):
  out = []
  for state_ix in range(len(game.game_states)):
    next_state = replace(game.game_states[state_ix], status=[], guesses=[])
    from_game = replace(game, game_states=game.game_states[:state_ix] + [next_state])
    guesses = generate_guess_sequence(
      from_game, model, beam_searcher, n_iterations, n_to_return)
    out.append(guesses)
  return out


def generate_guesses_from_queue(rank, devices, model, queue, bs, out_q, max_guesses, top_n,
                                batch_size=8, unk_boost=None):
  device = devices[rank]
  logging.info(f"Staring worker {rank}")
  if rank == 0:
    logging.info(f"Loading model {model}")
  model = load_model(model, device)
  _set_unk_boost(model, unk_boost, device)
  bs = bs.get_beam_searcher(model)
  new_games_available = True
  on_games = []
  on_guesses = []
  while new_games_available or len(on_games) > 0:
    # If our current pool of games is less than `batch_size` and
    # there are more games get from the queue, pull games from the queue
    while new_games_available and len(on_games) < batch_size:
      try:
        game = queue.get(block=False)
        on_games.append(game)
        on_guesses.append([])
      except Empty:
        new_games_available = False

    if len(on_games) == 0:
      # No games left to run on
      return

    # Run on the current batch of games
    new_guesses = generate_guess_batch(on_games, model, bs)
    next_games = []
    next_guesses = []
    for i in range(len(new_guesses)):
      all_guesses = on_guesses[i]
      all_guesses.append(new_guesses[i][:top_n])
      next_guess = new_guesses[i][0].phrase
      game = on_games[i]
      if next_guess == game.game_phrase or len(all_guesses) == max_guesses:
        # Done guesses for this game
        out_q.put((game.id, len(game.game_states)-1, all_guesses))
      else:
        # Need to generate another guess for this game
        cur_state = game.game_states[-1]
        status = [2 if w == p else 0 for w, p in zip(next_guess, game.game_phrase)]
        cur_state = replace(
          cur_state, guesses=cur_state.guesses + [next_guess], status=cur_state.status + [status])
        game = replace(game, game_states=game.game_states[:-1] + [cur_state])
        next_games.append(game)
        next_guesses.append(all_guesses)
    on_games = next_games
    on_guesses = next_guesses


def _set_unk_boost(model, val, device):
  if not val:
    return
  train_games = IconaryDataset("train").load()
  voc = set()
  for game in train_games:
    voc.update(model.tokenizer.encode(" ".join(game.game_phrase), add_special_tokens=False))
  voc_size = model.model.config.vocab_size
  is_known = np.zeros(voc_size)
  is_known[list(voc)] = -val
  is_known = torch.as_tensor(is_known, dtype=torch.float32, device=device)
  model.register_buffer("generation_bias", is_known)
  return model


def generate_from_models(
    model, ds, config: GuessingConfig, output_file,
    batch_size=1, unk_boost=None
):
  """Generate guesses from game states in human/human games

  :param model: Model directory to run from
  :param ds: IconaryDataset to generate from
  :param config: Generation configuration
  :param output_file: Where to save the output
  :param batch_size: Batch size of generation beam search
  :param unk_boost: Amount to boost unknown words
  :return: evaluation resutls
  """
  if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()
    logging.info(f"Found {n_gpus} GPUs")
  else:
    logging.info("No GPU found, using CPU")
    device = torch.device("cpu")
    n_gpus = 0

  games = ds.load()

  if output_file is not None:
    logging.info(f"Will save output into {output_file}")

  tasks = []
  predictions = {}
  for game in games:
    predictions[game.id] = [[] for _ in range(len(game.game_states))]
    for state_ix in range(len(game.game_states)):
      cur_state = replace(game.game_states[state_ix], guesses=[], status=[])
      tasks.append(replace(game, game_states=game.game_states[:state_ix]+[cur_state]))

  # Might improve performance a bit to keep similar length games in the same batch
  tasks.sort(key=lambda x: (len(x.game_phrase), x.id))

  if n_gpus <= 1:
    model = load_model(model, device)
    _set_unk_boost(model, unk_boost, device)
    bs = config.beam_search.get_beam_searcher(model)
    pbar = tqdm(total=len(tasks), desc="gen", ncols=100)
    while tasks:
      batch = tasks[-batch_size:]
      tasks = tasks[:-batch_size]
      guesses = generate_guess_batch(batch, model, bs)
      for guess, game in zip(guesses, batch):
        guess = guess[:config.top_n_to_save]
        cur_state = game.game_states[-1]
        # print(game.id, len(predictions[game.id]), len(game.game_states))
        predictions[game.id][len(game.game_states)-1].append(guess)
        next_guess = guess[0].phrase
        if next_guess == game.game_phrase or len(cur_state.guesses) == config.n_guesses_per_state-1:
          # Done guessing for this game state
          pbar.update(1)
        else:
          # Queue up another guess
          status = [2 if w == p else 0 for w, p in zip(next_guess, game.game_phrase)]
          cur_state = replace(
            cur_state, guesses=cur_state.guesses + [next_guess], status=cur_state.status + [status])
          input_game = replace(game, game_states=game.game_states[:-1] + [cur_state])
          tasks.append(input_game)
    pbar.close()
  else:
    ctx = torch.multiprocessing.get_context("spawn")

    logging.info(f"Splitting up generation on {n_gpus} devices")
    q = ctx.Queue()
    pbar = tqdm(desc="gen", ncols=100, total=len(tasks))
    for game in tasks:
      q.put(game)
    out_q = ctx.Queue()

    args = (list(range(n_gpus)), model, q, config.beam_search, out_q,
            config.n_guesses_per_state, config.top_n_to_save, batch_size, unk_boost)
    context = torch.multiprocessing.spawn(
      generate_guesses_from_queue, nprocs=n_gpus, args=args, join=False)

    while pbar.total != pbar.n:
      game_id, n, guesses = out_q.get()
      predictions[game_id][n] = guesses
      pbar.update(1)

    while not context.join():
      pass

  logging.info("Evaluating...")
  results = eval_guesses(games, predictions, ds.is_ood)
  logging.info(json.dumps(results, indent=2))

  if output_file is not None:
    logging.info(f"Saving output into {output_file}")
    save_guesses(config, ds, predictions, output_file, results)


def main():
  parser = argparse.ArgumentParser(
    description="Generate guesses the model would have made for game states in human/human games")
  parser.add_argument("model")
  parser.add_argument("--output_file", help="Where to save the guesses")
  parser.add_argument("--beam_size", type=int, default=20)
  parser.add_argument("--max_steps", type=int, default=30)
  parser.add_argument("--unk_boost", type=float, default=None,
                      help="Rare word boosting amount, we generally use 2.0")
  parser.add_argument("--n_guesses_per_state", type=int, default=5)
  parser.add_argument("--dataset", choices=IconaryDataset.SPLITS, default="ind-valid", nargs="+",
                      help="Which dataset(s) to run on")
  parser.add_argument("--batch_size", type=int, default=12)
  parser.add_argument("--top_n", type=int, default=None)
  parser.add_argument("--sample", type=int, default=None, help="Subsample the test data")

  args = parser.parse_args()

  utils.add_stdout_logger()

  logging.info("Loading generation config")
  config = GuessingConfig(
    top_n_to_save=args.top_n,
    n_guesses_per_state=args.n_guesses_per_state,
    beam_search=AllenNLPBeamSearcher(args.beam_size, None, args.max_steps)
  )

  model = utils.select_run_dir(args.model)

  logging.info("Generating from config: " + str(config))

  for dataset in args.dataset:
    ds = IconaryDataset(dataset, sample=args.sample)
    logging.info(f"Evaluating on dataset {ds.get_name()}")
    generate_from_models(
      model, ds, config, args.output_file,
      batch_size=args.batch_size, unk_boost=args.unk_boost
    )


if __name__ == '__main__':
  main()

import logging
from collections import defaultdict
from typing import Dict, List

from iconary.data.datasets import IconaryGame, get_train_voc
from iconary.models.generation import Guess
import numpy as np


def get_soft_win_leeway(phrase_len):
  if phrase_len <= 2:
    return 0
  if phrase_len <= 5:
    return 1
  return 2


def eval_game(game: IconaryGame, game_predictions: List, eval_oov=True) -> Dict[str, float]:
  """Evaluates a prediction on our win and soft-win metrics.

  :param game: IconaryGame to evaluate on
  :param game_predictions: For each state in the game, for each guess in that state, either
                           a list of tokens or `Guess` object, OR a list of such objects
                           in which case the first one will be used.
  :param eval_oov: Are we evaluating OOV games
  :return:
  """
  if eval_oov:
    voc = get_train_voc()
  if len(game.game_states) != len(game_predictions):
    raise ValueError(f"Game {game.id} has {len(game_predictions)} state "
                     f"predictions but {len(game.game_states)} states.")
  known = game.is_given
  guessed = list(known)
  for state_ix, state_pred in enumerate(game_predictions):
    for guess in state_pred:
      if isinstance(guess, list):  # Guess is a list
        # Assume guess is an n-best less and select the first (best) option
        if isinstance(guess[0], (list, Guess)):
          guess = guess[0]

      if isinstance(guess, Guess):
        guess = guess.phrase

      if len(guess) > len(game.game_phrase):
        logging.warning(
          f'Guess phrase {" ".join(guess)} was longer than phrase {" ".join(game.game_phrase)}')
        guess_tok = guess[:len(game.game_phrase)]
      else:
        guess_tok = guess

      for i, w in enumerate(guess_tok):
        if w == game.game_phrase[i] and not known[i]:
          guessed[i] = True

    # update what words are known given the previous states
    for human_guess in game.game_states[state_ix].guesses:
      for i, w in enumerate(human_guess):
        if w == game.game_phrase[i]:
          known[i] = True

  if eval_oov:
    oov = np.mean([g for i, g in enumerate(guessed) if game.game_phrase[i] not in voc])
    leeway = get_soft_win_leeway(len(game.game_phrase))
    missing = sum(not x for x in guessed)
    return dict(win=all(guessed), oov=oov, soft_win=oov > 0 and missing <= leeway)
  else:
    leeway = get_soft_win_leeway(len(game.game_phrase))
    missing = sum(not x for x in guessed)
    return dict(win=all(guessed), soft_win=missing <= leeway)


def eval_guesses(games: List[IconaryGame], predictions: Dict[str, List],
                 eval_oov: bool, max_guesses=5) -> Dict[str, float]:
  """
  Compute automatic metrics guesses on game states in a corpus of human/human
  """
  for game_guesses in predictions.values():
    for i, state_guesses in enumerate(game_guesses):
      game_guesses[i] = state_guesses[:max_guesses]

  out = defaultdict(list)
  for g in games:
    for k, v in eval_game(g, predictions[g.id], eval_oov).items():
      out[k].append(v)
  return {k: np.mean(v) for k, v in out.items()}
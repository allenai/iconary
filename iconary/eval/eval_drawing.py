from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict

from iconary.data.datasets import IconaryIcon, CANVAS_HEIGHT, INITIAL_ICON_WIDTH, CANVAS_WIDTH, \
  IconaryGame
from iconary.models.generation import GeneratedDrawing
import numpy as np

from iconary.utils import utils


@dataclass
class DrawingPredictions:
  """
  Predicted Drawings for a corpus for games. Since many games can have the same initial game
  phrase, we store the initial drawings for the phrase in one dictionary, and drawings
  for subsequent games states in a second dictionary
  """

  """Phrase to an initial drawing for that phrase"""
  phrase_to_drawings: Dict[str, List[GeneratedDrawing]]

  """game_id to drawings for games states after the initial one"""
  game_id_to_drawings: Dict[str, List[List[GeneratedDrawing]]]

  """game_id to initial phrase"""
  game_id_to_initial_phrase: Dict[str, str]


def is_valid(icon: IconaryIcon, minscaling, minwidth, aspect_ratio) -> bool:
  if icon.scale_factor <= minscaling:
    return False
  if icon.x + 0.5 * icon.width <= minwidth:
    return False
  if icon.y + 0.5 * icon.height <= minwidth * aspect_ratio:
    return False
  if icon.x - 0.5 * icon.width >= 1 - minwidth:
    return False
  if icon.y - 0.5 * icon.height >= 1 - minwidth * aspect_ratio:
    return False
  return True


def _is_valid(icon: IconaryIcon) -> bool:
  """
  Checks if the icon is essentially invsiible, used to clean up both human and AI
  drawings during evaluation (humans could create such icons using our UI).
  :param icon: Icon to Check
  :return: True if the icon should be used in evaluation
  """
  aspect_ratio = CANVAS_WIDTH / CANVAS_HEIGHT
  minwidth = 1.0 * INITIAL_ICON_WIDTH / (2 * CANVAS_WIDTH)
  minscaling = 0.1
  return is_valid(icon, minscaling, minwidth, aspect_ratio)


def score_drawing(given: List[IconaryIcon], references: List[List[IconaryIcon]]):
  """Score a single drawing

  :param given: The proposed drawings
  :param references: List of ground truth drawings
  :return: icon-f1 and icon-f1-median
  """
  names = Counter(x.name for x in given if _is_valid(x))
  scores = []
  for ref in references:
    ref_names = Counter(x.name for x in ref if _is_valid(x))
    u = sum((names | ref_names).values())
    i = sum((names & ref_names).values())
    if u != 0:
      scores.append(i/u)
    else:
      scores.append(0)
  scores = np.array(scores)
  return scores.max(), np.median(scores)


def eval_drawings(
    games: List[IconaryGame], references: List[IconaryGame],
    drawing_prediction: DrawingPredictions
) -> Dict[str, float]:
  """
  Evaluate drawings for a corpus of human/human games

  :param games: Games we have drawings for
  :param references: Games to compare to when evaluating, this can include more games than in
                     `games` since, if two games share the same starting game phrase, we use
                     both games when evaluating results for either one of those games.
  :param drawing_prediction: The predictions
  :return: dictionary of metrics
  """
  initial_phrase_refs = defaultdict(list)
  for game in references:
    initial_phrase_refs[" ".join(game.game_phrase)].append(game.game_states[0].drawing)

  all_scores = defaultdict(list)
  for game in games:
    initial_phrase = " ".join(game.game_phrase)
    first_drawing = drawing_prediction.phrase_to_drawings[initial_phrase]
    first_drawing = first_drawing[0].icons
    all_scores[game.id].append(score_drawing(first_drawing, initial_phrase_refs[initial_phrase]))

    next_drawings = drawing_prediction.game_id_to_drawings[game.id]
    for i, (drawing, state) in enumerate(zip(next_drawings, game.game_states[1:])):
      drawing = drawing[0].icons
      ref = state.drawing
      all_scores[game.id].append(score_drawing(drawing, [ref]))

  initial_drawing_score = np.array([x[0] for x in all_scores.values()])
  subsequent_scores = utils.flatten_list(x[1:] for x in all_scores.values())
  if len(subsequent_scores) == 0:
    subsequent_scores = None
  else:
    subsequent_scores = np.array(subsequent_scores)
  names = ["icon-f1", "icon-f1-median"]

  out = {}
  for kind, scores in [(None, initial_drawing_score), ("subsequent", subsequent_scores)]:
    if scores is None:
      for name in names:
        out[name] = 0
    else:
      for name, score in zip(names, scores.mean(0)):
        if kind is None:
          out[f"{name}"] = score
        else:
          out[f"{kind}-{name}"] = score
  return out

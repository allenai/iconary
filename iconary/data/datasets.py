import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from os import listdir
from os.path import join, exists, dirname
from typing import Any, List, Dict, Union, Tuple, Optional

from allennlp.common import Registrable

from iconary import file_paths
from iconary.utils import utils
from iconary.utils.utils import cache_from_s3, load_json_object, dump_json_object
import numpy as np


INITIAL_ICON_WIDTH = 80
INITIAL_ICON_HEIGHT = 80
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 450

GIVEN_WORDS = {'it', 'and', 'his', 'of', 'their', '_', 'an', 'a', 'the', 'her'}


"""Specify everything we know so far about a game phrase. For each token in the phrase,
this is either a str (indicating the correct word) or a list of words (indicating 
 words that are known to be incorrect)."""
GamePhraseConstraints = List[Union[str, List[str]]]


@dataclass
class IconaryIcon:
  """An Icon in an Iconary drawing"""

  @staticmethod
  def build_icon(icon_name_or_id, mirror, rotation, scale, x, y, icon_num=None,
                 w=None, h=None):
    if w is None:
      width = INITIAL_ICON_WIDTH * scale / CANVAS_WIDTH
    else:
      width = w
    if h is None:
      height = INITIAL_ICON_HEIGHT * scale / CANVAS_HEIGHT
    else:
      height = h
    if isinstance(icon_name_or_id, int):
      icon_name = get_icon_names()[icon_name_or_id]
    else:
      icon_name = icon_name_or_id
    if icon_num is not None:
      icon_id = '_'.join([icon_name, datetime.now().isoformat('T'), '%02d' % icon_num])
    else:
      icon_id = '_'.join([icon_name, datetime.now().isoformat('T')])
    return IconaryIcon(icon_id, mirror, icon_name_or_id, rotation, scale, width, height, x, y)

  """ID for this icon, unique to the game its in"""
  id: str
  mirrored: bool
  name: str
  rotation_degrees: float
  scale_factor: float

  """Width normalized by the canvas width to be between 0 and 1"""
  width: float

  """Height normalized by the canvas height"""
  height: float

  """Icon's center x coordinate normalized by canvas width"""
  x: float

  """Icon's center y coordinate normalized by canvas height"""
  y: float

  def to_dict(self):
    return dict(
      id=self.id,
      mirrored=self.mirrored,
      name=self.name,
      rotation_degrees=self.rotation_degrees,
      scale_factor=self.scale_factor,
      width=self.width,
      height=self.height,
      x=self.x,
      y=self.y
    )

  def get_bounding_box(self):
    w = self.width / 2
    h = self.height / 2
    return self.x - w, self.y - h, self.x + w, self.y + h


@dataclass(frozen=True)
class IconaryRound:
  """A drawing and sequences of guesses in an Iconary game"""

  """The drawing"""
  drawing: Union[List[IconaryIcon], Any]

  """Tokenized guesses made after this drawing"""
  guesses: List[List[str]]

  """0 for incorrect, 1 for close, 2 for correct"""
  status: List[List[int]]

  meta: Any = None

  def to_dict(self):
    return dict(
      drawing=[x.to_dict() for x in self.drawing],
      guesses=list(self.guesses),
      status=list(self.status)
    )


@dataclass(frozen=True)
class IconaryGame:
  """A full Iconary game"""

  """Unique ID"""
  id: str

  """Tokenized target game phrase"""
  game_phrase: List[str]

  """Which game phrase words are given at test time"""
  is_given: List[bool]

  """Game rounds, can be empty"""
  game_states: List[IconaryRound]

  meta: Any = None

  def __post_init__(self):
    assert len(self.is_given) == len(self.game_phrase)
    assert not all(self.is_given)

  @property
  def known_words(self) -> List[str]:
    return ["_" if given else w for given, w in zip(self.game_phrase, self.is_given)]

  @staticmethod
  def from_dict(obj: Dict[str, Any]) -> 'IconaryGame':
    states = []
    for state in obj["game_states"]:
      icons = []
      for icon in state["drawing"]:
        icons.append(IconaryIcon(
          icon["id"], icon["mirrored"], icon["name"], icon["rotation_degrees"],
          icon["scale_factor"], icon["width"], icon["height"], icon["x"], icon["y"]
        ))
      states.append(IconaryRound(icons, state["guesses"], state["status"]))

    if "given_words" in obj:
      is_given = [x != "_" for x in obj["given_words"]]
    else:
      is_given = obj["is_given"]
    return IconaryGame(
      obj["id"], obj["game_phrase"], is_given,
      states, obj["meta"])

  def to_dict(self) -> Dict[str, Any]:
    dictionary = dict(
      id=self.id,
      game_phrase=list(self.game_phrase),
      is_give=list(self.is_given),
      meta=self.meta,
      game_states=[x.to_dict() for x in self.game_states]
    )
    return dictionary

  def get_known_words(self) -> List[bool]:
    """Return which game phrase words are known after all rounds so far"""
    known = list(self.is_given)
    for state in self.game_states:
      for guess, status in zip(state.guesses, state.status):
        for i, (w, s) in enumerate(zip(guess, status)):
          if s == 2:
            known[i] = True
    return known

  @staticmethod
  def matches_constraints(constraints: GamePhraseConstraints, sent: List[str]) -> bool:
    """Checks if `sent` meets `constraints`"""
    if len(constraints) != len(sent):
      return False

    for const, w in zip(constraints, sent):
      if isinstance(const, str):
        if const != w:
          return False
      elif w in const:
        return False
    return True

  def get_constraints(self, state_ix=None, guess_ix=None) -> GamePhraseConstraints:
    """Return constrains on the game phrase known so far"""

    # Use dictionary so order is preserved
    constraints = [w if g else {} for g, w in zip(self.is_given, self.game_phrase)]
    for state in self.game_states[:state_ix]:
      for guess in state.guesses[:guess_ix]:
        for i, (w, target_w) in enumerate(zip(guess, self.game_phrase)):
          if w == "_":
            continue
          if w == target_w:
            constraints[i] = w
          elif isinstance(constraints[i], dict):
            constraints[i][w] = None
          else:
            pass  # Must have guessed a word that (should have) alreay known
    return [x if isinstance(x, str) else list(x) for x in constraints]

  def get_last_guess(self) -> Optional[List[str]]:
    """Return the most recent guess"""
    for state in self.game_states[::-1]:
      if len(state.guesses) > 0:
        return state.guesses[-1]
    return None


def convert_json_point(data: Dict) -> IconaryGame:
  game_states = []
  for gs in data["game_states"]:
    icons = []
    for icon in gs["drawing"]:
      icons.append(IconaryIcon(
        icon["id"], icon["mirrored"], icon["name"], icon["rotation_degrees"],
        icon["scale_factor"], icon["width"], icon["height"], icon["x"], icon["y"]
      ))
    game_states.append(IconaryRound(icons, gs["guesses"], gs["status"]))

  if len(game_states[0].drawing) != 0 or len(game_states[0].guesses) != 1:
    raise RuntimeError()
  return IconaryGame(
    data["id"], data["game_phrase"], game_states[0].guesses[0], game_states[1:])


def sample_num_to_str(sample):
  if sample % 1000 == 0:
    return str(sample // 1000) + "k"
  else:
    return str(sample)


def sample_from_list(x, sample, sample_seed=None):
  if sample is None:
    return x
  if sample_seed is None:
    sample_seed = 69851
  np.random.RandomState(sample_seed).shuffle(x)
  if isinstance(sample, int):
    return x[:sample]
  else:
    return x[sample[0]:sample[1]]


class Dataset(Registrable):
  """Generic super-class for datasets"""

  def get_name(self):
    raise NotImplementedError()

  def load(self) -> List[IconaryGame]:
    raise NotImplementedError()


def load_from_json(f):
  with open(f, "r") as fh:
    data = json.load(fh)
  return [IconaryGame.from_dict(x) for x in data["games"]]


def load_games(split):
  if split not in {"train", "ind-valid", "ood-valid"}:
    raise ValueError()
  file_name = f"{split}.json"
  file = join(file_paths.ICONARY_HOME, file_name)
  s3_path = f"public-datasets/{file_name}"
  cache_from_s3("ai2-vision-iconary", s3_path, file)
  return load_from_json(file)


@Dataset.register("iconary")
class IconaryDataset(Dataset):

  SPLITS = {"ind-valid", "ood-valid", "train", "ood-test", "ind-test"}

  def __init__(self, split: str, sample: Optional[int] = None):
    if split not in self.SPLITS:
      raise ValueError(f"Unknown split {split} vaid splits are {self.SPLITS}")
    ds_name = f"inconary-{split}"
    if sample is not None:
      ds_name += f"-s{sample_num_to_str(sample)}"
    self.ds_name = ds_name
    self.split = split
    self.sample = sample

  @property
  def is_ood(self):
    return "ood" in self.split

  def get_name(self):
    return self.ds_name

  def load(self) -> List[IconaryGame]:
    games = load_games(self.split)
    if self.sample is not None:
      games.sort(key=lambda x: x.id)
      rng = np.random.RandomState(931314)
      rng.shuffle(games)
      games = games[:self.sample]
    return games


def get_train_voc():
  source = file_paths.TRAIN_VOC_CACHE
  if exists(source):
    return load_json_object(source)
  logging.info("Cache train voc...")
  tr = IconaryDataset("train").load()
  tr = sorted(set(utils.flatten_list(x.game_phrase for x in tr)))
  dump_json_object(tr, source)
  logging.info("Done")
  return tr


_ICON_2_ID = None


def get_icon_names():
  """Returns the list of all possible icon names"""
  src = join(dirname(__file__), "icon_list.json")
  global  _ICON_2_ID
  if _ICON_2_ID is None:
    icon_data = load_json_object(src)
    icon_names = [x["name"] for x in icon_data]
    _ICON_2_ID = icon_names
  return _ICON_2_ID


import datetime
import re
from collections import Counter, OrderedDict, defaultdict
from typing import List

from allennlp.common import Registrable, FromParams
from dataclasses import replace
import numpy as np

from iconary.data.datasets import IconaryIcon, IconaryGame, get_icon_names
from iconary.utils import utils


class Drawing2Text(Registrable):
  """Converts a drawing to plain text"""

  def encode_drawing(self, drawing: List[IconaryIcon]) -> str:
    raise NotImplementedError()

  def get_special_tokens(self):
    return None

  def get_name(self):
    raise ValueError()


def encode_drawings(drawing_encoder: Drawing2Text, game: IconaryGame):
  encoded_states = []
  for game_state in game.game_states:
    encoded_states.append(
      replace(game_state, drawing=drawing_encoder.encode_drawing(game_state.drawing)))
  return replace(game, game_states=encoded_states)


REMAP_NAMES = {
  "arrow_straight": "arrow",
  "curved arrow2": "curved arrow",
  "curved arrow": "curved arrow",
}


def get_icon_tokens(icon_name, icon_token_cache={}):
  if icon_name in icon_token_cache:
    return list(icon_token_cache[icon_name])
  tokens = REMAP_NAMES.get(icon_name, icon_name).replace("_", " ").split()
  icon_token_cache[icon_name] = tokens
  return list(tokens)


@Drawing2Text.register("icon-list")
class IconList(Drawing2Text):
  def __init__(self, max_icons=None):
    self.max_icons = max_icons

  def get_name(self):
    return f"icon-list"

  def encode_drawing(self, drawing: List[IconaryIcon]):
    if self.max_icons:
      drawing = prune_drawing(drawing, self.max_icons)
    return " ".join(utils.flatten_list(get_icon_tokens(x.name) for x in drawing))

  def _float_to_text(self, x):
    return str(int(x * self.factor))


@Drawing2Text.register("null-drawer")
class NullDrawer(Drawing2Text):
  def get_name(self):
    return f"null"

  def encode_drawing(self, drawing: List[IconaryIcon]):
    return ""


@Drawing2Text.register("icon-counts")
class IconCounts(Drawing2Text):
  def __init__(self, max_unique_icons=None, num_first=True, comma=False, sort=None):
    self.max_unique_icons = max_unique_icons
    self.comma = comma
    self.num_first = num_first
    self.sort = sort

  def get_name(self):
    return f"icon-counts"

  def encode_drawing(self, drawing: List[IconaryIcon]):
    if self.sort == "x0":
      drawing = sorted(drawing, key=lambda x: x.x - x.width / 2.0)
    elif self.sort == "x":
      drawing = sorted(drawing, key=lambda x: x.x)
    elif self.sort == "area":
      drawing = sorted(drawing, key=lambda x: x.width*x.height)
    elif self.sort is not None:
      raise NotImplementedError()

    # Counts retain input order, so the sort will be preserved
    counts = Counter(x.name for x in drawing)
    if self.max_unique_icons and self.max_unique_icons > len(counts):
      name_to_pos = {}
      for i, k in enumerate(drawing):
        if k.name not in name_to_pos:
          name_to_pos[k.name] = i
      items = [(k, c, name_to_pos[k]) for k, c in counts.items()]
      items.sort(key=lambda x: x[1:])
      pruned_counts = Counter()
      for name, _, _ in items[:self.max_unique_icons]:
        pruned_counts[name] = counts[name]
      counts = pruned_counts

    out = []
    for i, (k, v) in enumerate(counts.items()):
      icon_tokens = get_icon_tokens(k)
      if v > 1:
        if self.num_first:
          icon_tokens = [str(v)] + icon_tokens
        else:
          icon_tokens.append(str(v))
      if i > 0:
        out[-1] = out[-1] + ","
      out += icon_tokens

    return " ".join(out)

  def _float_to_text(self, x):
    return str(int(x * self.factor))


def get_icon_modifiers(drawing: List[IconaryIcon]):
  median_scale = np.median([x.scale_factor for x in drawing])
  out = []
  for icon in drawing:
    modifiers = []
    if abs(icon.scale_factor > median_scale * 1.5) > 0:
      modifiers.append("huge")
    elif abs(icon.scale_factor > median_scale * 1.2) > 0:
      modifiers.append("large")
    elif abs(icon.scale_factor < median_scale * 0.8) > 0:
      modifiers.append("small")
    elif abs(icon.scale_factor < median_scale * 0.5) > 0:
      modifiers.append("tiny")

    if icon.name == "arrow_straight":
      rot = icon.rotation_degrees
      if 125 < rot < 145:
        modifiers.append("down")
      elif 305 < rot < 325:
        modifiers.append("up")
      elif rot < 135 or rot > 325:
        modifiers.append("right")
      else:
        modifiers.append("left")
    else:
      if 20 < icon.rotation_degrees < 340:
        modifiers.append("rotated")

    out.append(modifiers)
  return out


def get_icon_modifiers_v3(drawing: List[IconaryIcon]):
  median_scale = np.median([x.scale_factor for x in drawing])
  out = []
  for icon in drawing:
    modifiers = []
    if abs(icon.scale_factor > median_scale * 1.5) > 0:
      modifiers.append("huge")
    elif abs(icon.scale_factor > median_scale * 1.2) > 0:
      modifiers.append("large")
    elif abs(icon.scale_factor < median_scale * 0.8) > 0:
      modifiers.append("small")
    elif abs(icon.scale_factor < median_scale * 0.5) > 0:
      modifiers.append("tiny")

    if icon.name == "arrow_straight":
      rot = icon.rotation_degrees
      if icon.mirrored:
        rot = (rot + 270) % 360
      if 0 <= rot < 90:
        modifiers.append("right")
      elif 90 <= rot < 180:
        modifiers.append("down")
      elif 180 <= rot < 270:
        modifiers.append("left")
      else:
        modifiers.append("up")
    else:
      if 20 < icon.rotation_degrees < 340:
        modifiers.append("rotated")

    out.append(modifiers)
  return out


@Drawing2Text.register("icon-names")
class IconNames(Drawing2Text):
  def get_name(self):
    return f"icon-names"

  def encode_drawing(self, drawing: List[IconaryIcon]):
    icon_names = list(set(x.name for x in drawing))
    np.random.shuffle(icon_names)
    for i in range(len(icon_names) - 1):
      icon_names[-1] += ","
    return icon_names


@Drawing2Text.register("icon-counts-with-modifiers")
class IconCountsWithModifiers(Drawing2Text):
  def __init__(self, max_unique_icons=None, encoding_version=2,
               comma=True, order=None, group_icon_thresh=2):
    self.max_unique_icons = max_unique_icons
    self.encoding_version = encoding_version
    self.comma = comma
    self.group_icon_thresh = group_icon_thresh
    self.order = order
    if encoding_version not in {2, 3}:
      raise NotImplementedError()

  def get_name(self):
    return f"icon-counts-with-modifiers"

  def encode_drawing(self, drawing: List[IconaryIcon]):
    if len(drawing) == 0:
      return ""
    counts = defaultdict(list)
    if self.encoding_version == 3:
      mods = get_icon_modifiers_v3(drawing)
    else:
      raise NotImplementedError(self.encoding_version)

    for icon, modifiers in zip(drawing, mods):
      name = modifiers
      name.append(icon.name)
      counts[tuple(name)].append(icon)

    group = []
    for k, v in counts.items():
      if len(v) >= self.group_icon_thresh:
        group.append((k, v))
      else:
        for x in v:
          group.append((k, [x]))

    if self.max_unique_icons and self.max_unique_icons > len(counts):
      raise NotImplementedError()

    if self.order == "x":
      group = sorted(group, key=lambda x: min(i.x for i in x[1]))
    elif self.order == "x0":
      group = sorted(group, key=lambda x: min(i.x - (i.width / 2.0) for i in x[1]))
    elif self.order == "alpha":
      group = sorted(group, key=lambda x: x[1][0].name)
    elif self.order is None:
      pass
    else:
      raise NotImplementedError(self.order)

    if self.group_icon_thresh > 2:
      rebuilt_group = []
      chunk_key, chunk_icons = group[0]
      for k, v in group[1:]:
        if k == chunk_key:
          chunk_icons += v
        else:
          rebuilt_group.append((chunk_key, chunk_icons))

    out = []
    for k, v in group:
      v = len(v)
      if v > 1:
        out.append(str(v))
      out += list(k[:-1])
      out += get_icon_tokens(k[-1])
      if self.comma:
        out[-1] += ","
    if self.comma:
      out[-1] = out[-1][:-1]
    return " ".join(out)


def prune_drawing(icons, max_icons):
  if len(icons) > max_icons:
    counts = OrderedDict()
    for i, icon in enumerate(icons):
      if icon.name not in counts:
        counts[icon.name] = [i]
      else:
        counts[icon.name].append(i)
    n_to_remove = len(icons)- max_icons
    to_remove = []
    for i in range(n_to_remove):
      k = max(counts.items(), key=lambda x: (len(x[1]), x[1][-1]))[0]
      to_remove.append(counts[k].pop())
      if len(counts[k]) == 0:
        del counts[k]

    to_remove = set(to_remove)
    out = []
    for i, k in enumerate(icons):
      if i in to_remove:
        continue
      out.append(k)
    if len(out) != max_icons:
      raise ValueError()
    return out
  else:
    return icons


def original_prune_drawing(drawing: List[IconaryIcon], clip, max_icons):
  if clip is None:
    return drawing[:max_icons]
  clipped_drawing = []
  counter = 0
  for icon in drawing:
    if counter == clip and icon.name == clipped_drawing[-1].name:
      continue
    if counter == 0 or icon.name == clipped_drawing[-1].name:
      counter += 1
      clipped_drawing.append(icon)
    else:
      counter = 1
      clipped_drawing.append(icon)
  return clipped_drawing[:max_icons]


@Drawing2Text.register("drawing-encoder")
@Drawing2Text.register("original-drawing-encoder")
class DrawingEncoder(Drawing2Text):

  def __init__(
      self,
      canvas_width,
      canvas_height,
      num_x_location_bins,
      num_y_location_bins,
      xy_top_left,
      icon_size,
      num_rotation_buckets,
      scale_bucket_boundaries,
      icon_dictionary,
      clipping,
      max_icons,
      sort=None
  ):
    self.icon_size = icon_size
    self.canvas_width = canvas_width
    self.canvas_height = canvas_height
    self.num_x_location_bins = num_x_location_bins
    self.num_y_location_bins = num_y_location_bins
    self.xy_top_left = xy_top_left
    self.scale_bucket_boundaries = scale_bucket_boundaries
    self.icon_width = self.icon_height = icon_size
    self.num_rotation_buckets = num_rotation_buckets
    self.icon_dictionary = icon_dictionary
    self.max_icons = max_icons
    self.clipping = clipping
    self.sort = sort

    rot_bin_width = 360 / self.num_rotation_buckets
    self.shift = rot_bin_width / 2
    self.rotation_bins = np.array(
      rot_bin_width * np.arange(0, self.num_rotation_buckets), dtype=np.int32)
    loc_bin_width = 1.0 / self.num_x_location_bins
    loc_bin_height = 1.0 / self.num_y_location_bins
    self.location_bins_x = np.array(
      loc_bin_width * (np.arange(0, self.num_x_location_bins) + 0.5),
      dtype=np.float32
    )
    self.location_bins_y = np.array(
      loc_bin_height * (np.arange(0, self.num_y_location_bins) + 0.5),
      dtype=np.float32
    )
    self.scale_bins = np.array(scale_bucket_boundaries)
    self.lut = (self.scale_bins[:-1] + self.scale_bins[1:]) * 0.5

    if self.icon_dictionary:
      icon_names = get_icon_names()
      self.icon_ix = {k: i for i, k in enumerate(icon_names)}
      self.id2icon = icon_names
    else:
      self.icon_ix = None

  def construct_drawing(self, tokens):
    icons = []
    for i in range(len(tokens) // 6):
      icon_tokens = tokens[i*6:(i+1)*6]
      icon_id = int(icon_tokens[0][5:-1])
      icon_name = self.id2icon[icon_id]

      x = int(icon_tokens[1][2:-1])
      x = np.asscalar(self.location_bins_x[x])

      y = int(icon_tokens[2][2:-1])
      y = np.asscalar(self.location_bins_y[y])

      scale = int(icon_tokens[3][6:-1])
      scale = self.lut[scale]

      rot = int(icon_tokens[4][4:-1])
      rot = np.asscalar(self.rotation_bins[rot])

      mirrored = icon_tokens[5] == "[MIRROR1]"

      icons.append(IconaryIcon.build_icon(icon_name, mirrored, rot, scale, x, y, i))
    return icons

  def get_special_token_order(self):
    return ["ICON", "X", "Y", "SCALE", "ROT", "MIRROR"]

  def get_token(self, kind, bin):
    return f"[{kind}{bin}]"

  def get_special_token_counts(self):
    special_token_counts = {
      "MIRROR": 2,
      "ROT": self.num_rotation_buckets,
      "X": self.num_x_location_bins,
      "Y": self.num_y_location_bins,
      "SCALE": len(self.scale_bins) - 1
    }
    if self.icon_dictionary:
      special_token_counts["ICON"] = len(self.icon_ix)
    return special_token_counts

  def get_special_tokens(self):
    all_special_tokens = []
    counts = self.get_special_token_counts()
    for k in self.get_special_token_order():
      all_special_tokens += [self.get_token(k, i) for i in range(counts[k])]
    return all_special_tokens

  def encode_drawing(self, drawing: List[IconaryIcon]):
    drawing = original_prune_drawing(drawing, self.clipping, self.max_icons)
    if self.sort == "x":
      drawing.sort(key=lambda icon: (icon.x - icon.width /2, icon.y - icon.height /2))
    elif self.sort == "alpha":
      drawing.sort(key=lambda icon: (icon.name, icon.x - icon.width /2, icon.y - icon.height /2))
    elif self.sort == "area":
      drawing.sort(key=lambda icon: (-icon.height*icon.width, icon.name, icon.x - icon.width /2, icon.y - icon.height /2))
    elif self.sort == "custom1":
      def order(icon: IconaryIcon):
        if "arrow" in icon.name:
          return (1, -(icon.x - icon.width /2))
        area = icon.height*icon.width
        x, y = icon.x - icon.width / 2, icon.y - icon.height / 2
        return (-1, -area, -x, -y)
      drawing.sort(key=order)
    elif self.sort is not None:
      raise NotImplementedError(self.sort)
    return " ".join(utils.flatten_list(self.encode_icon(x) for x in drawing))

  def encode_icon(self, icon: IconaryIcon):
    tokens = {}

    tokens["MIRROR"] = int(icon.mirrored)

    shifted = (self.shift + icon.rotation_degrees)
    if shifted > 360:
      shifted -= 360
    quantized_rotation = np.floor(shifted / (360 / self.num_rotation_buckets))
    quantized_rotation = int(min(max(quantized_rotation, 0), self.num_rotation_buckets - 1))
    tokens["ROT"] = quantized_rotation

    tokens["X"] = int(max(min(icon.x * self.num_x_location_bins, self.num_x_location_bins - 1), 0))
    tokens["Y"] = int(max(min(icon.y * self.num_y_location_bins, self.num_y_location_bins - 1), 0))

    current_scale = int(
      np.digitize([icon.scale_factor], self.scale_bins)[0]) - 1
    current_scale = min(current_scale, len(self.scale_bins) - 1 - 1)
    tokens["SCALE"] = current_scale

    if self.icon_dictionary:
      out = [self.get_token("ICON", self.icon_ix[icon.name])]
    else:
      out = get_icon_tokens(icon.name)
    for kind in self.get_special_token_order()[1:]:
      out.append(self.get_token(kind, tokens[kind]))
    return out


@Drawing2Text.register("coordinate-str")
class CoordinateString(Drawing2Text):

  def __init__(self, places, mode="center", max_icons=None, coordinate_sep=":",
               modifiers=None):
    self.places = places
    self.mode = mode
    self.modifiers = modifiers
    self.coordinate_sep = coordinate_sep
    self.factor = (10 ** self.places)
    self.max_icons = max_icons

  def get_name(self):
    return f"coordinate-str-{self.places}"

  def encode_drawing(self, drawing: List[IconaryIcon]):
    if len(drawing) == 0:
      return []
    if self.max_icons is not None:
      drawing = prune_drawing(drawing, self.max_icons)
    if self.modifiers is not None:
      modifiers = get_icon_modifiers_v3(drawing)

    tokens = []
    for ix, icon in enumerate(drawing):
      if self.modifiers:
        tokens += modifiers[ix]
      tokens += get_icon_tokens(icon.name)
      if self.coordinate_sep:
        tokens[-1] += self.coordinate_sep
      tokens += self.get_coordinate_tokens(icon)
      if ix != len(drawing) - 1:
        tokens[-1] += ","
    return tokens

  def _float_to_text(self, x):
    return str(int(x * self.factor))

  def get_coordinate_tokens(self, icon: IconaryIcon):
    tokens = []
    if self.mode == "center":
      tokens += [
        self._float_to_text(icon.x),
        self._float_to_text(icon.y),
      ]
    else:
      tokens += [
        self._float_to_text(icon.x - icon.width / 2),
        self._float_to_text(icon.y - icon.height / 2),
        self._float_to_text(icon.x + icon.width / 2),
        self._float_to_text(icon.y + icon.height / 2)
      ]
    return tokens


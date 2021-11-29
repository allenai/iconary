import warnings
from typing import Union, List

from allennlp.common import Registrable
from dataclasses import dataclass
from transformers import T5Tokenizer

from iconary.data.datasets import IconaryGame
from iconary.models.constraints_to_text import ConstraintsToText
from iconary.models.drawing_to_text import Drawing2Text, encode_drawings


class T5GameToText(Registrable):
  """Converts a Iconary game to input/output text"""

  def get_special_tokens(self):
    return []

  def init_special_tokens(self, tokenizer, t5_embedding):
    return None

  def game_to_text(self, game: IconaryGame, get_constraints=False):
    raise NotImplementedError()

  def post_process_generation(self, tokenizer, output_ids, game):
    raise NotImplementedError()


@T5GameToText.register("guess-phrase")
class GuessPhrase(T5GameToText):
  def __init__(
      self,
      constraint_encoder: ConstraintsToText,
      drawing_prefix: Union[str, List], drawing_encoder: Drawing2Text,
      end_with_period=True, n_drawings=1,
      constraints_first=True,
  ):
    self.constraint_encoder = constraint_encoder
    self.drawing_prefix = drawing_prefix
    self.drawing_encoder = drawing_encoder
    self.end_with_period = end_with_period
    self.n_drawings = n_drawings
    self.constraints_first = constraints_first

  def get_special_tokens(self):
    return self.drawing_encoder.get_special_tokens()

  def game_to_text(self, game: IconaryGame, get_constraints=False):
    game = encode_drawings(self.drawing_encoder, game)
    input_text = []
    if self.constraints_first:
      input_text += self.constraint_encoder.build_text(game.get_constraints())
      if self.end_with_period and len(input_text) > 0:
        input_text[-1] += "."
    else:
      input_text = []

    for state in game.game_states[-self.n_drawings:]:
      if self.drawing_prefix is not None:
        if isinstance(self.drawing_prefix, str):
          input_text.append(self.drawing_prefix)
        else:
          input_text += self.drawing_prefix
      input_text += state.drawing
      if self.end_with_period:
        input_text[-1] += "."

    if not self.constraints_first:
      constraint_text = self.constraint_encoder.build_text(game.get_constraints())
      if self.end_with_period and len(constraint_text) > 0:
        constraint_text[-1] += "."
      input_text += constraint_text

    if get_constraints:
      return input_text, game.game_phrase, game.get_constraints()
    else:
      return input_text, game.game_phrase

  def post_process_generation(self, tokenizer, output_ids, game):
    return tokenizer.decode(output_ids).split()


@T5GameToText.register("predict-phrases-bart")
class PredictPhraseBART(T5GameToText):
  def __init__(
      self,
      drawing_prefix: Union[str, List],
      drawing_encoder: Drawing2Text,
      phrase_prefix: List[str],
      n_drawings=1,
      end_with_period=True,
      output="phrase"
  ):
    self.drawing_prefix = drawing_prefix
    self.drawing_encoder = drawing_encoder
    self.phrase_prefix = phrase_prefix
    self.n_drawings = n_drawings
    self.end_with_period = end_with_period
    self.output = output

  def get_special_tokens(self):
    self.drawing_encoder.get_special_tokens()

  def game_to_text(self, game: IconaryGame, get_constraints=False):
    game = encode_drawings(self.drawing_encoder, game)
    input_text = []
    for state in game.game_states[-self.n_drawings:]:
      if self.drawing_prefix is not None:
        if isinstance(self.drawing_prefix, str):
          input_text.append(self.drawing_prefix)
        else:
          input_text += self.drawing_prefix
      input_text += state.drawing
      if len(input_text) > 0:
        input_text[-1] += "."

    input_text += self.phrase_prefix

    phrase = game.game_phrase
    constraints = game.get_constraints()

    if self.end_with_period:
      phrase = phrase[:-1] + [phrase[-1] + "."]
      if isinstance(constraints[-1], str):
        constraints[-1] += "."

    if self.output == "phrase":
      output_text = phrase
      output_constraints = constraints
    elif self.output == "all":
      output_constraints = input_text + constraints
      output_text = list(input_text) + phrase
    else:
      raise NotImplementedError()

    on_blank = False
    for i in range(len(constraints)):
      if isinstance(constraints[i], list):
        if not on_blank:
          input_text.append("<mask>")
          on_blank = True
      else:
        on_blank = False
        input_text.append(constraints[i])

    if get_constraints:
      return input_text, output_text, output_constraints
    else:
      return input_text, output_text

  def post_process_generation(self, tokenizer, output_ids, game):
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    if self.end_with_period and text[-1] == ".":
      text = text[:-1]
    text = text.split(" ")
    if self.output == "phrase":
      return text
    else:
      if len(self.phrase_prefix) != 1:
        raise NotImplementedError()
      for i in range(len(text)):
        if text[i] == self.phrase_prefix[0]:
          out = text[i+1:]
          missing = len(game.game_phrase) - len(out)
          if missing > 0:
            # Generation hit max_seq_len and never finished the output phrase
            out += game.given_words[-missing:]
          if len(out) != len(game.game_phrase):
            return game.given_words

          return out

      warnings.warn("missing some required output tokens, max_seq_len might be too short")
      return game.given_words


@T5GameToText.register("fill-in-phrase-blanks")
class FillInPhraseBlanks(T5GameToText):

  def __init__(
      self,
      drawing_prefix: Union[str, List],
      drawing_encoder: Drawing2Text,
      phrase_prefix: List[str],
      n_drawings=1,
      end_with_period=True,
  ):
    self.drawing_prefix = drawing_prefix
    self.drawing_encoder = drawing_encoder
    self.phrase_prefix = phrase_prefix
    self.n_drawings = n_drawings
    self.end_with_period = end_with_period

  def get_special_tokens(self):
    self.drawing_encoder.get_special_tokens()

  def game_to_text(self, game: IconaryGame, get_constraints=False):
    game = encode_drawings(self.drawing_encoder, game)
    input_text = []
    for state in game.game_states[-self.n_drawings:]:
      if self.drawing_prefix is not None:
        if isinstance(self.drawing_prefix, str):
          input_text.append(self.drawing_prefix)
        else:
          input_text += self.drawing_prefix
      input_text.append(state.drawing + ".")

    constraints = game.get_constraints()

    input_text += self.phrase_prefix

    output_text = []
    output_constraint = []
    span_start = None
    on_token = 0
    phrase = game.game_phrase
    if self.end_with_period:
      phrase = phrase[:-1] + [phrase[-1] + "."]

    for w_i in range(len(phrase)+1):
      if w_i == len(phrase) or isinstance(constraints[w_i], str):
        if span_start is not None:
          # Found a span of unknown words
          special_token = f"<extra_id_{on_token}>"
          input_text.append(special_token)

          output_text.append(special_token)
          output_text += phrase[span_start:w_i]

          output_constraint.append(special_token)
          output_constraint += constraints[span_start:w_i]

          on_token += 1

          span_start = None
        if w_i != len(phrase):
          input_text.append(phrase[w_i])
      else:
        if span_start is None:
          span_start = w_i

    input_text = " ".join(input_text)
    output_text = " ".join(output_text)
    if get_constraints:
      return input_text, output_text, output_constraint
    else:
      return input_text, output_text

  def post_process_generation(self, tokenizer, output_ids, game):
    phrase_const = game.get_constraints()
    out = []
    token_id = 32099
    eos_id = tokenizer.eos_token_id
    on_word_piece = 0
    for i, const in enumerate(phrase_const):
      if isinstance(const, list):
        # Token `i` is unconstrained, so the next token in the output_ids should generate it
        if i == len(phrase_const)-1 or isinstance(phrase_const[i + 1], str):
          if on_word_piece >= (len(output_ids)-1) or output_ids[on_word_piece] != token_id:
            if on_word_piece >= len(output_ids)-1:
              # We hit the max sequence length while generating so did not build text
              # for this output. Give up and just return blank tokens
              out += ["_" for _ in range(len(phrase_const) - len(out))]
              return out
            else:
              # Otherwise something went very wrong
              raise RuntimeError(f"Expected special token {token_id} in the output")
          token_id -= 1
          start = on_word_piece + 1
          on_word_piece += 2
          while (on_word_piece+1 < len(output_ids) and
                 output_ids[on_word_piece] != token_id and
                 output_ids[on_word_piece] != eos_id):
            on_word_piece += 1
          out += tokenizer.decode(output_ids[start:on_word_piece]).split()
      else:
        # Token `i` is given
        out.append(const)
    if self.end_with_period and out[-1].endswith("."):
      out[-1] = out[-1][:-1]
    return out

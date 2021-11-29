from typing import List, Optional

from allennlp.common import Registrable, Params
from allennlp.nn.beam_search import BeamSearch, Sampler
from dataclasses import asdict, dataclass, replace

from iconary.data.datasets import IconaryIcon, IconaryGame
from iconary.utils.utils import load_json_object


@dataclass
class BeamSearchConfig(Registrable):

  def get_beam_searcher(self, model):
    # TODO or we could just have generate(model, games).....
    raise NotImplementedError()


@dataclass
@BeamSearchConfig.register("allennlp-beam-search")
class AllenNLPBeamSearcher(BeamSearchConfig):
  beam_size: int
  per_node_beam_size: Optional[int]
  max_steps: int
  sampler: Optional[Sampler]=None

  def get_beam_searcher(self, model):
    return BeamSearch(
      model.get_eos(), self.max_steps, self.beam_size, self.per_node_beam_size, self.sampler)


@dataclass
class GeneratedDrawing:
  icons: List[IconaryIcon]
  sequence_output: List[str]
  logprob: float

  @staticmethod
  def from_dict(x):
    return GeneratedDrawing(
      [IconaryIcon(**i) for i in x["icons"]], x["sequence_output"], x["logprob"])


def generate_drawing(game: IconaryGame, model, bs) -> List[GeneratedDrawing]:
  start_pred, start_state, step_fn = model.initialize_decoding([game])
  predictions, log_probs = bs.search(start_pred, start_state, step_fn)
  out = []
  for pred, conf in zip(predictions[0].cpu(), log_probs[0].cpu()):
    sequence_out, drawing = model.construct_drawing(pred)
    out.append(GeneratedDrawing(drawing, sequence_out, float(conf)))
  return out


def generate_drawings(games: List[IconaryGame], model, bs, top_n=None) -> List[List[GeneratedDrawing]]:
  start_pred, start_state, step_fn = model.initialize_decoding(games)
  predictions, log_probs = bs.search(start_pred, start_state, step_fn)
  out = []
  if top_n is not None:
    predictions = predictions[:, :top_n]
  else:
    predictions = predictions[:, :1]

  for batch in range(len(games)):
    game_out = []
    for pred, conf in zip(predictions[batch].cpu(), log_probs[batch].cpu()):
      sequence_out, drawing = model.construct_drawing(pred)
      game_out.append(GeneratedDrawing(drawing, sequence_out, float(conf)))
    out.append(game_out)
  return out


@dataclass
class Guess:
  phrase: List[str]
  confidence: float
  valid: Optional[bool]


def guesses_to_dictionary(data):
  """Nested structure of `Guess` objects -> Nested structure of dictionaries"""
  if isinstance(data, Guess):
    return asdict(data)
  elif isinstance(data, list):
    return [guesses_to_dictionary(x) for x in data]
  elif isinstance(data, dict):
    return {k: guesses_to_dictionary(v) for k, v in data.items()}
  else:
    raise NotImplementedError()


def dictionary_to_guesses(data):
  """Nested structure of dictionaries -> Nested structure of `Guess` objects"""
  if isinstance(data, dict) and set(data.keys()) == {"phrase", "confidence", "valid"}:
    return Guess(**data)
  elif isinstance(data, list):
    return [dictionary_to_guesses(x) for x in data]
  elif isinstance(data, dict):
    return {k: dictionary_to_guesses(v) for k, v in data.items()}
  else:
    raise NotImplementedError()


def load_guesses(input_file):
  data = load_json_object(input_file)
  return dictionary_to_guesses(data["guesses"])


def generate_guess(game: IconaryGame, model, bs) -> List[Guess]:
  start_pred, start_state, step_fn = model.initialize_decoding([game])
  predictions, log_probs = bs.search(start_pred, start_state, step_fn)

  const = game.get_constraints()
  out = []
  invalid = []
  for pred, conf in zip(predictions[0].cpu(), log_probs[0].cpu()):
    tokens = model.post_process_generation(pred, game)
    guess = Guess(tokens, float(conf), IconaryGame.matches_constraints(const, tokens))
    if guess.valid:
      out.append(guess)
    else:
      invalid.append(guess)
  return out + invalid


def generate_guess_batch(games: List[IconaryGame], model, bs) -> List[List[Guess]]:
  start_pred, start_state, step_fn = model.initialize_decoding(games)
  predictions, log_probs = bs.search(start_pred, start_state, step_fn)

  batch_out = []
  for game, game_pred, game_logprob in zip(games, predictions.cpu(), log_probs.cpu()):
    const = game.get_constraints()
    out = []
    invalid = []
    for pred, conf in zip(game_pred, game_logprob):
      tokens = model.post_process_generation(pred, game)
      guess = Guess(tokens, float(conf), IconaryGame.matches_constraints(const, tokens))
      if guess.valid:
        out.append(guess)
      else:
        invalid.append(guess)
    batch_out.append(out + invalid)
  return batch_out


def generate_guess_sequence(
    game: IconaryGame, model, beam_searcher, max_guesses, n_to_return=1):
  all_guesses = []
  for i in range(max_guesses):
    guesses = generate_guess(game, model, beam_searcher)[:n_to_return]
    all_guesses.append(guesses)
    next_guess = guesses[0].phrase
    if next_guess == game.game_phrase:
      return all_guesses
    status = [2 if w == p else 0 for w, p in zip(game.game_phrase, next_guess)]
    cur_state = game.game_states[-1]
    cur_state = replace(
      cur_state, guesses=cur_state.guesses + [next_guess],
      status=cur_state.status + [status])
    game = replace(game, game_states=game.game_states[:-1] + [cur_state])

  return all_guesses


def generate_from_human_guesses(game: IconaryGame, model, beam_searcher, n_to_return=1):
  guesses = []
  from_state = game.game_states[-1]
  for guess_ix in range(len(from_state.guesses)):
    cur_state = replace(
      from_state, guesses=from_state.guesses[:guess_ix], status=from_state.status[:guess_ix])
    from_game = replace(game, game_states=game.game_states[:-1] + [cur_state])
    guess = generate_guess(from_game, model, beam_searcher)[:n_to_return]
    guesses.append(guess)
  return guesses


from collections import defaultdict, Counter
from typing import List

from allennlp.common import Registrable, FromParams
from dataclasses import replace

from torch.utils.data import Dataset as TorchDataset, Sampler
import numpy as np

from iconary.data.datasets import IconaryGame


class ImsituGameIterator(Registrable):
  """
  During training, we want models to see part of each completed game (i.e. The first drawing
  and the first `n` guesses) rather then the entire game, which could already be solved
  by the human player. These iterators control which parts are selected to be trained on.
  """
  def build_dataset(self, examples) -> TorchDataset:
    raise NotImplementedError()

  def is_batch_iterator(self):
    return False


@ImsituGameIterator.register("first-drawing-no-guesses")
class FirstDrawingNoGuessesIterator(ImsituGameIterator):
  def build_dataset(self, examples: List[IconaryGame]):
    data = []
    for ex in examples:
      first_state = replace(ex.game_states[0], guesses=[], status=[])
      data.append(replace(ex, game_states=[first_state]))
    return data


class AllDrawingsRandomGuessDataset(TorchDataset):
  def __init__(self, examples: List[IconaryGame]):
    self.examples = examples
    self.candidates = []
    for game_ix, ex in enumerate(examples):
      self.candidates += list((game_ix, state_ix) for state_ix in range(len(ex.game_states)))

  def __getitem__(self, item):
    game_ix, state_ix = self.candidates[item]
    game = self.examples[game_ix]
    on_state = game.game_states[state_ix]
    guesses, status = on_state.guesses, on_state.status
    while len(guesses) > 0 and all(s == 2 for s in status[-1]):
      guesses = guesses[:-1]
      status = status[:-1]
    if len(guesses) > 0:
      ix = np.random.randint(0, len(guesses) + 1)
      guesses = guesses[:ix]
      status = status[:ix]
    new_state = replace(on_state, guesses=guesses, status=status)
    new_states = game.game_states[:state_ix] + [new_state]
    return replace(game, game_states=new_states)

  def __len__(self):
    return len(self.candidates)


@ImsituGameIterator.register("all-drawings-random-guess")
class AllDrawingsRandomGuess(ImsituGameIterator):
  def build_dataset(self, examples: List[IconaryGame]):
    return AllDrawingsRandomGuessDataset(examples)


class WithHumanGuessesDataset(TorchDataset):
  def __init__(self, examples: List[IconaryGame], human_guess_w):
    self.examples = examples
    self.human_guess_w = human_guess_w
    self.candidates = []
    for game_ix, ex in enumerate(examples):
      for state_ix, state in enumerate(ex.game_states):
        self.candidates.append((game_ix, state_ix, False))
        if len(state.guesses) >= 1 and any(s != 2 for s in state.status[0]):
          self.candidates.append((game_ix, state_ix, True))

  def __getitem__(self, item):
    game_ix, state_ix, human_guess = self.candidates[item]
    game = self.examples[game_ix]
    on_state = game.game_states[state_ix]
    guesses, status = on_state.guesses, on_state.status
    while len(guesses) > 0 and all(s == 2 for s in status[-1]):
      guesses = guesses[:-1]
      status = status[:-1]
    if human_guess:
      if len(guesses) == 0:
        raise ValueError()
      if len(guesses) > 1:
        ix = np.random.randint(0, len(guesses))
      else:
        ix = 0
      new_state = replace(on_state, guesses=guesses[:ix], status=status[:ix])
      new_states = game.game_states[:state_ix] + [new_state]
      return replace(game, game_states=new_states, weight=self.human_guess_w,
                     meta=dict(human_guess_target=guesses[ix]))
    else:
      if len(guesses) > 0:
        ix = np.random.randint(0, len(guesses) + 1)
        guesses = guesses[:ix]
        status = status[:ix]
      new_state = replace(on_state, guesses=guesses, status=status)
      new_states = game.game_states[:state_ix] + [new_state]
      return replace(game, game_states=new_states, weight=1.0)

  def __len__(self):
    return len(self.candidates)


@ImsituGameIterator.register("random-guess-with-human-phrases")
class WithHumanGuesses(ImsituGameIterator):
  def __init__(self, human_guess_w):
    self.human_guess_w = human_guess_w

  def build_dataset(self, examples: List[IconaryGame]):
    return WithHumanGuessesDataset(examples, self.human_guess_w)


class HumanGuessesDataset(TorchDataset):
  def __init__(self, examples: List[IconaryGame]):
    self.examples = examples
    self.candidates = []
    for game_ix, ex in enumerate(examples):
      for state_ix, state in enumerate(ex.game_states):
        if len(state.guesses) >= 1:
          self.candidates.append((game_ix, state_ix))

  def __getitem__(self, item):
    game_ix, state_ix = self.candidates[item]
    game = self.examples[game_ix]
    on_state = game.game_states[state_ix]
    guesses, status = on_state.guesses, on_state.status

    if len(guesses) > 1:
      ix = np.random.randint(0, len(guesses))
    elif len(guesses) == 1:
      ix = 0
    else:
      raise RuntimeError()

    new_state = replace(on_state, guesses=guesses[:ix], status=status[:ix])
    new_states = game.game_states[:state_ix] + [new_state]
    return replace(game, game_states=new_states)

  def __len__(self):
    return len(self.candidates)


@ImsituGameIterator.register("random-human-guess")
class HumanGuesses(ImsituGameIterator):
  def __init__(self):
    pass

  def build_dataset(self, examples: List[IconaryGame]):
    return HumanGuessesDataset(examples)


class AllDrawingsPreviousGuessDataset(TorchDataset):
  def __init__(self, examples: List[IconaryGame]):
    self.examples = examples
    self.candidates = []
    for game_ix, ex in enumerate(examples):
      for state_ix in range(len(ex.game_states)):
        const = replace(ex, game_states=ex.game_states[:state_ix]).get_constraints()
        if not all(isinstance(x, str) for x in const):
          self.candidates.append((game_ix, state_ix))

  def __getitem__(self, item):
    game_ix, state_ix = self.candidates[item]
    game = self.examples[game_ix]
    last_state = replace(game.game_states[state_ix], guesses=[], status=[])
    return replace(game, game_states=game.game_states[:state_ix] + [last_state])

  def __len__(self):
    return len(self.candidates)


@ImsituGameIterator.register("all-drawings-previous-guess")
class AllDrawingsPreviousGuess(ImsituGameIterator):
  def build_dataset(self, examples: List[IconaryGame]):
    return AllDrawingsPreviousGuessDataset(examples)

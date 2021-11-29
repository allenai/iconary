from allennlp.common import Registrable


class GamePhraseToText(Registrable):
  """Maps the game phrase and constraints to text (for the Drawer)"""
  def build_text(self, game_phrase, constraints) -> str:
    raise NotImplementedError()


@GamePhraseToText.register("game-phrase")
class GamePhrase(GamePhraseToText):

  def build_text(self, game_phrase, constraints) -> str:
    return " ".join(game_phrase)


@GamePhraseToText.register("game-phrase-and-constraints")
class GamePhraseAndConstraints(GamePhraseToText):

  def build_text(self, game_phrase, constraints) -> str:
    tokens = []
    for (word, const) in zip(game_phrase, constraints):
      if const == word:
        tokens.append(word)
        tokens.append("(known)")
      else:
        if not isinstance(const, list):
          raise RuntimeError()
        tokens.append(word)
        tokens.append("(tried")
        tokens += const
        tokens[-1] += ")"
    return " ".join(tokens)


@GamePhraseToText.register("game-phrase-mark-known")
class GamePhraseMarkKnown(GamePhraseToText):
  def __init__(self, mode="star", mark_known=True):
    self.mode = mode
    self.mark_known = mark_known

  def build_text(self, game_phrase, constraints):
    tokens = []
    for (word, const) in zip(game_phrase, constraints):
      if self.mark_known and word != const:
        tokens.append(word)  # Unmarked word
      # Else mark the word
      elif self.mode == "paran":
        tokens.append("(" + word + ")")
      elif self.mode == "star":
        tokens.append(word + "*")
      elif self.mode is None:
        tokens.append(word)
      else:
        raise NotImplementedError(self.mode)
    return " ".join(tokens)

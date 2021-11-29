from allennlp.common import FromParams, Registrable

from iconary.data.datasets import GamePhraseConstraints


class ConstraintsToText(Registrable):
  """Maps game-phrase constraints to plain text"""
  def get_special_tokens(self):
    return []

  def build_text(self, constraints:  GamePhraseConstraints) -> str:
    raise NotImplementedError()


@ConstraintsToText.register("null-constraints")
class NullConstraints(ConstraintsToText):
  def build_text(self, constraints:  GamePhraseConstraints):
    return ""


@ConstraintsToText.register("known-words")
class KnownWords(ConstraintsToText):
  def __init__(self, prefix="", blank_token=None):
    self.blank_token = blank_token
    self.prefix = prefix
    assert isinstance(self.prefix, str)

  def build_text(self, constraints:  GamePhraseConstraints):
    return self.prefix + " " + " ".join([w if isinstance(w, str) else self.blank_token for w in constraints])


@ConstraintsToText.register("exclude-set-and-known")
class ExcludeSetAndKnown(ConstraintsToText):
  def __init__(self, exclude_prefix="exclude:", given_prefix="given:"):
    self.given_prefix = given_prefix
    self.exclude_prefix = exclude_prefix
    self.blank = None

  def build_text(self, constraints:  GamePhraseConstraints):
    known_words = []
    exclude_words = set()
    valid_words = set()
    for const in constraints:
      if isinstance(const, str):
        known_words.append(const)
        valid_words.add(const)
      else:
        known_words.append("_")
        exclude_words.update(const)

    exclude_words = exclude_words.difference(valid_words)
    input_text = [self.exclude_prefix] + list(sorted(exclude_words))
    input_text[-1] = input_text[-1] + "."
    input_text += [self.given_prefix]
    input_text += known_words
    known_words[-1] = known_words[-1] + "."
    return " ".join(input_text)


@ConstraintsToText.register("constraint-parans")
class ConstraintParans(ConstraintsToText):
  def __init__(self, prefix=None, always_include_blank=True, blank_token=None,
               max_constraints=None):
    self.prefix = prefix
    self.max_constraints = max_constraints
    self.always_include_blank = always_include_blank
    self.blank_token = blank_token
    self.blank = blank_token

  def build_text(self, constraints: GamePhraseConstraints):
    input_text = [self.prefix] if self.prefix is not None else []
    for const in constraints:
      if isinstance(const, str):
        input_text.append(const)
      else:
        if self.max_constraints:
          const = const[-self.max_constraints:]
        if self.always_include_blank or len(const) == 0:
          input_text.append(self.blank)
        if len(const) > 0:
          input_text.append("(not")
          for w in const[:-1]:
            input_text.append(w + ",")
          input_text.append(const[-1] + ")")
    return " ".join(input_text)

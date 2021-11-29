from typing import Any, Callable, List, Dict, Tuple, Union

import torch
from allennlp.common import Registrable, Params
from allennlp.nn.beam_search import StepFunctionType
from dataclasses import replace, dataclass
from torch import nn

from iconary.data.datasets import Dataset, IconaryGame


class IconaryModel(nn.Module, Registrable):
  """Model that takes IconaryGames as input"""

  def collate(self, games: List[IconaryGame]) -> Dict:
    """Converts a list of pre-processed games to tensor input for `self.forward`"""
    raise NotImplementedError()

  def collate_train(self, games: List[IconaryGame]) -> Dict:
    """A version of `collate` to use during training"""
    return self.collate(games)

  def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes training (loss, logits, labels) from the output for `self.collate`"""
    raise NotImplementedError()

  # Generation methods used for decoding
  def initialize_decoding(self, game: List[IconaryGame]) -> Tuple[torch.Tensor, Dict, StepFunctionType]:
    """Returns elements we can use in allennlp's BeamSearch"""
    raise NotImplementedError()

  def get_eos(self):
    raise NotImplementedError()

  def get_pad(self):
    raise NotImplementedError()

  def post_process_generation(self, output_ids, game):
    """Transform generation output into a text guess we can use in the game"""
    raise NotImplementedError()


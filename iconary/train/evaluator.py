from collections import OrderedDict

import torch
from allennlp.common import Registrable
from torch.nn import functional as F

from iconary.utils import utils


class Evaluator(Registrable):
  """Evalutors get fed a series of batches and with model predictions, during which some
  statistics are accumulated and can be retrieved with `get_stats`"""

  def clear(self):
    raise NotImplementedError()

  def get_stats(self):
    raise NotImplementedError()

  def evaluate_batch(self, predictions):
    raise NotImplementedError()

  def preprocess(self, datasets):
    """
    Give the evaluator a chance see that data before it is used, we need this
    since `ClfHardEasyEvaluator` needs to pre-load some dataset statistics before use
    """
    pass


@Evaluator.register("clf-evaluator")
class ClfEvaluator(Evaluator):

  def __init__(self, accuracy=True, nll=True, prefix_format="{metric}"):
    super().__init__()
    self.accuracy = accuracy
    self.nll = nll
    self.prefix_format = prefix_format
    self._stats = OrderedDict()

  def clear(self):
    self._stats = OrderedDict()

  def get_stats(self):
    out = {}
    for k, v in self._stats.items():
      out[k] = v[0] / float(v[1])
    return out

  def evaluate_batch(self, predictions):
    for k, v in self.get_batch_stats(predictions).items():
      if isinstance(v, torch.Tensor):
        v = utils.numpy(v)
      cur = self._stats.get(k)
      if cur is None:
        self._stats[k] = v
      else:
        self._stats[k] = (cur[0]+v[0], cur[1]+v[1])

  def get_batch_stats(self, model_output):
    if isinstance(model_output, tuple):
      logits, labels = model_output[1:3]
      logits = F.log_softmax(logits, -1)
    else:
      logits = model_output.logprobs
      labels = model_output.labels

    if len(logits.size()) > 2:
      logits = logits.view(-1, logits.size(-1))
      labels = labels.view(-1)

    valid = (labels >= 0).float()
    valid_sum = valid.sum()
    result = OrderedDict()
    if self.accuracy:
      prefix = self.prefix_format.format(metric="acc")
      correct = (torch.argmax(logits, 1) == labels).float()
      result[prefix] = (correct.dot(valid), valid_sum)
    if self.nll:
      ix = torch.arange(len(logits), dtype=torch.long, device=labels.device)
      prefix = self.prefix_format.format(metric="nll")
      nll = -logits[ix, labels]
      result[prefix] = (nll.dot(valid), valid_sum)
    return result

import logging
from typing import List

import torch
from allennlp.common import Params
from torch import nn
from transformers import T5Tokenizer, AutoConfig, T5ForConditionalGeneration, AutoTokenizer

from iconary.data.datasets import IconaryGame
from iconary.models.drawing_to_text import DrawingEncoder, get_icon_tokens, encode_drawings
from iconary.models.game_phrase_to_text import GamePhraseToText
from iconary.models.game_to_text import T5GameToText
from iconary.models.iconary_model import IconaryModel
from iconary.models.t5_decoding import t5_initialize_decoding, t5_decoding_token_constraints
from iconary.utils import utils, ops
from iconary.utils.utils import to_device
from torch.nn import functional as F, CrossEntropyLoss
import numpy as np


def t5_construct_labels(decoder_input_ids, decoder_attention_mask):
  labels = decoder_attention_mask * decoder_input_ids - 100 * (1 - decoder_attention_mask)
  labels = labels[:, 1:].contiguous()
  return labels


@IconaryModel.register("tguesser")
@IconaryModel.register("t5-text2text")
class TGuesser(IconaryModel):

  @classmethod
  def from_params(
        cls,
        params: Params,
        constructor_to_call=None,
        constructor_to_inspect=None,
        **extras,
    ):
      # Some backwards compatibility fixes for older version of this class
      if "fixed_shared" in params:
        params.pop("fixed_shared")
      if "initialize_from" in params:
        assert params.pop("initialize_from") is None
      return super().from_params(params, constructor_to_call, constructor_to_inspect)

  def __init__(
      self,
      pretrained_model: str,
      game_to_text: T5GameToText,
      freeze_embed=False,
      dont_generate_banned_words=True
  ):
    super().__init__()
    self.freeze_embed = freeze_embed
    self.pretrained_model = pretrained_model
    self.game_to_text = game_to_text
    self.dont_generate_banned_words = dont_generate_banned_words

    special_tokens = self.game_to_text.get_special_tokens()
    self.special_tokens = special_tokens
    self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_model, verbose=False)
    self.special_token_offset = len(self.tokenizer)
    if special_tokens:
      self.tokenizer.tokenizer.add_tokens(special_tokens)

    self.model = T5ForConditionalGeneration.from_pretrained(self.pretrained_model)
    embed = self.model.resize_token_embeddings(len(self.tokenizer))
    if special_tokens:
      embed[-len(special_tokens):] = 0

    if self.freeze_embed:
      self.model.shared.weight.requires_grad = False

  @property
  def seq_len(self):
    return self.model.config.n_positions

  def initialize(self):
    pass

  def collate(self, games: List[IconaryGame]):
    input_text, output_text = utils.transpose_lists([self.game_to_text.game_to_text(g) for g in games])
    cond_text = self.tokenizer(input_text, return_tensors='pt', padding=True, max_length=self.seq_len)
    out_text = self.tokenizer(output_text, return_tensors='pt', padding=True, max_length=self.seq_len)

    cond_text["labels"] = out_text["input_ids"]
    return cond_text

  def forward(self, input_ids, attention_mask, labels):
    # Dont evaluate the trailing pad tokens
    labels = torch.where(labels == 0, torch.full_like(labels, -100), labels)
    loss, logits = self.model(input_ids, attention_mask, labels=labels)[:2]
    return loss, logits, labels

  def get_eos(self):
    return self.tokenizer.eos_token_id

  def get_pad(self):
    return self.tokenizer.pad_token_id

  def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
    if prefix + "generation_bias" in state_dict:
      generation_bias = state_dict.pop(prefix + "generation_bias")
    else:
      generation_bias = None

    # Previous versions optimized out huggingfaces's redundantly encoded embeddings from the
    # state dicts, we no longer to that, so add them back here for backwards compatibility
    if "model.encoder.embed_tokens.weight" not in state_dict:
      embed = state_dict["model.shared.weight"]
      for k in ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "model.lm_head.weight"]:
        state_dict[k] = embed

    super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs)
    if generation_bias is not None:
      self.register_buffer("generation_bias", generation_bias)

  def initialize_decoding(self, games: List[IconaryGame]):
    input_text, output_text, constraints = utils.transpose_lists(
      [self.game_to_text.game_to_text(g, True) for g in games])

    # TODO can we just use a non-tokenized representation?
    inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, max_length=self.seq_len)
    input_features = dict(inputs)

    for param in self.parameters():
      device = param.device
      break
    input_features = to_device(input_features, device)

    input_embeds = self.model.shared(input_features["input_ids"])
    encoder_out = self.model.encoder(
      inputs_embeds=input_embeds, attention_mask=input_features["attention_mask"])[0]

    if hasattr(self, "generation_bias") and self.generation_bias is not None:
      def post_process(model_out, model_inputs):
        return model_out.logits + self.generation_bias.unsqueeze(0).unsqueeze(1)
    else:
      post_process = None

    return t5_initialize_decoding(
      self.tokenizer, self.model, encoder_out,
      input_features["attention_mask"],
      constraints,
      prevent_known_incorrect_words=self.dont_generate_banned_words,
      post_process=post_process
    )

  def post_process_generation(self, output_ids, game):
    return self.game_to_text.post_process_generation(
      self.tokenizer, output_ids, game)


@IconaryModel.register("t5-basic-drawer")
@IconaryModel.register("tdrawer")
class TDrawer(IconaryModel):

  def __init__(
      self,
      pretrained_model: str,
      game_phrase_encoder: GamePhraseToText,
      drawing_encoder: DrawingEncoder,
      icon_name_init=True,
      number_init="count",
      train_with_constraints=False,
      freeze_word_embed=False,
      freeze_all_embed=False
  ):
    super().__init__()
    self.pretrained_model = pretrained_model
    self.game_phrase_encoder = game_phrase_encoder
    self.drawing_encoder = drawing_encoder
    self.icon_name_init = icon_name_init
    self.number_init = number_init
    self.train_with_constraints = train_with_constraints
    self.freeze_word_embed = freeze_word_embed
    self.freeze_all_embed = freeze_all_embed

    special_tokens = self.drawing_encoder.get_special_tokens()
    self.special_tokens = special_tokens

    self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, verbose=False)
    self.special_token_offset = len(self.tokenizer)
    self.tokenizer.add_tokens(special_tokens)

    self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(self.pretrained_model)
    embed = self.model.resize_token_embeddings(len(self.tokenizer))
    embed.weight.data[-len(special_tokens):] = 0

    if not drawing_encoder.icon_dictionary:
      raise NotImplementedError()
    special_token_order = self.drawing_encoder.get_special_token_order()
    constraints = torch.zeros(len(special_token_order), len(self.tokenizer), dtype=torch.bool)
    icon_kind_to_ix = {k: i for i, k in enumerate(special_token_order)}

    for token, token_ix in self.tokenizer.get_vocab().items():
      if token.startswith("[") and token.endswith("]"):
        token = token[1:-1]
        for kind, kind_ix in icon_kind_to_ix.items():
          if token.startswith(kind):
            constraints[kind_ix, token_ix] = True
    for i in range(6):
      if not torch.any(constraints[i]):
        raise ValueError(i)
    self.register_buffer("constraint_matrix", constraints)
    self.constraint_matrix[0, self.tokenizer.eos_token_id] = True  # Generate EOS

    if self.freeze_all_embed:
      self.model.shared.weight.requires_grad = False

    elif self.freeze_word_embed:
      # Freeze the word piece embeddings, but not the special token embeddings
      def freeze_word_embed_hook(grad):
        g = grad.clone()
        g[:32100] = 0
        return g
      self.model.shared.weight.register_hook(freeze_word_embed_hook)

  def initialize(self):
    if self.number_init or (self.icon_name_init and self.drawing_encoder.icon_dictionary):
      token_init = {}
      special_token_counts = self.drawing_encoder.get_special_token_counts()
      if self.icon_name_init and self.drawing_encoder.icon_dictionary:
        n_icon_init = []
        num_to_name = {v: k for k, v in self.drawing_encoder.icon_ix.items()}
        for i in range(special_token_counts["ICON"]):
          icon_name = num_to_name[i]
          # icon_token_ids = self.tokenizer.encode_tokens_with_cache(get_icon_tokens(icon_name))
          icon_token_ids = self.tokenizer.encode(
            " ".join(get_icon_tokens(icon_name)), add_special_tokens=False)
          n_icon_init.append(len(icon_token_ids))
          token_init[self.drawing_encoder.get_token("ICON", i)] = icon_token_ids

        n_icon_init = np.array(n_icon_init)
        logging.info(f"Initializing {len(n_icon_init)} icons with mean name "
                     f"embeddings (av {n_icon_init.mean():.2f} tokens)")

      if self.number_init is not None:
        for kind, count in special_token_counts.items():
          if kind == "ICON":
            continue
          for i in range(count):
            if self.number_init == "percent":
              init_str = str(int(round((i / count) * 100)))
            elif self.number_init == "count":
              init_str = str(i)
            else:
              raise NotImplementedError(self.number_init)
            init_ids = self.tokenizer.encode(init_str, add_special_tokens=False)
            token_init[self.drawing_encoder.get_token(kind, i)] = init_ids

      embed = self.model.shared
      embed_w = embed.weight.data
      n_initialized = 0
      for tok_ix, tok in enumerate(self.special_tokens):
        if tok not in token_init:
          continue
        n_initialized += 1
        embed_w[self.special_token_offset + tok_ix] = embed(torch.as_tensor(token_init[tok])).mean(0)

  def collate(self, data: List[IconaryGame], with_labels=True):
    for game in data:
      if all(isinstance(x, str) for x in game.get_constraints()):
        logging.warning("collating a game that is already solved")

    seq_len = self.model.config.n_positions

    input_text = [
      self.game_phrase_encoder.build_text(game.game_phrase, game.get_constraints())
      for game in data
    ]
    cond_text = self.tokenizer(input_text, return_tensors='pt', padding=True, max_length=seq_len)

    if with_labels:
      out_text = self.tokenizer(
        [self.drawing_encoder.encode_drawing(x.game_states[-1].drawing) for x in data],
        return_tensors='pt', padding=True, max_length=seq_len)
      cond_text["labels"] = out_text["input_ids"]
    return cond_text

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      labels=None,
      per_token_loss=False,
  ):
    labels = torch.where(labels == 0, torch.full_like(labels, -100), labels)
    lm_logits = self.model(input_ids, attention_mask, labels=labels, return_dict=True).logits

    if self.train_with_constraints or not self.training:
      if labels.size(1) % 6 != 1:  # 6 per icon and EOS
        raise ValueError()
      constraints = self.constraint_matrix.repeat(labels.size(1) // 6, 1)
      constraints = torch.cat([constraints, self.constraint_matrix[:1]], 0)
      constraints = constraints.unsqueeze(0)
      lm_logits = ops.mask_logits(lm_logits, constraints)

    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

    if per_token_loss:
      loss = F.cross_entropy(
        lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction="none")
      return loss.view(labels.size()), labels
    return loss, lm_logits, labels

  def initialize_decoding(self, games: List[IconaryGame]):
    input_features = self.collate(games, with_labels=False)
    for param in self.parameters():
      device = param.device
      break
    input_features = to_device(input_features, device)
    encoder_out = self.model.encoder(**input_features)[0]
    return t5_decoding_token_constraints(
      self.tokenizer, self.model, encoder_out,
      input_features["attention_mask"],
      self.constraint_matrix, constraints_before_softmax=True
    )

  def get_eos(self):
    return self.tokenizer.eos_token_id

  def get_pad(self):
    return self.tokenizer.pad_token_id

  def post_process_generation(self, output_ids):
    return self.tokenizer.decode(output_ids).split()

  def construct_drawing(self, output_ids):
    sequence_out = self.tokenizer.decode(output_ids).split()
    drawing = self.drawing_encoder.construct_drawing(sequence_out)
    return sequence_out, drawing

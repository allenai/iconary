"""Logic to use T5 in allennlp's BeamSearch

This allows us to use T5 to create the initial_state, starting predictions, and `StepFunctionType`
functions needed to run `BeamSearch` from allennlp. It also includes the constraint logic we use to
ensure Drawers produce valid drawings and Guesser produce valid guesses.

We use allennlp because it BeamSearch is easy to use and add constraints to, and hugging faces
version is much more difficult to modify.
"""

import torch
from transformers import T5Tokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

from iconary.data.datasets import GamePhraseConstraints
from iconary.utils import utils
from torch.nn import functional as F


_t5_tokens_with_spaces = None


def get_tokens_with_space(tokenizer):
  assert isinstance(tokenizer, T5Tokenizer)
  global _t5_tokens_with_spaces
  if _t5_tokens_with_spaces is None:
    start_of_word_tokens_ix = []
    for k, ix in tokenizer.get_vocab().items():
      if k.startswith("▁"):
        start_of_word_tokens_ix.append(ix)
    start_of_word_tokens_ix = torch.tensor(start_of_word_tokens_ix, dtype=torch.long)
    _t5_tokens_with_spaces = start_of_word_tokens_ix
  return _t5_tokens_with_spaces


def get_starts_word(tokenizer, voc_size=None):
  if voc_size is None:
    voc_size = len(tokenizer)
  start_of_word_tokens_ix = get_tokens_with_space(tokenizer)
  starts_new_word = torch.zeros(voc_size, dtype=torch.bool)
  starts_new_word[start_of_word_tokens_ix] = 1
  return starts_new_word


def build_allowed_word_matrix(
    constraints: GamePhraseConstraints, starts_new_word, continues_word, tokenizer):
  """
  :param constraints: from `ImsituGame.get_constraints()`
  :param starts_new_word: [vocab_size] tensor with 1 indicating word pieces that add a space
  :param tokenizer: T5 Tokenizer to use
  :return: A matrix of (number of spaces so far -> allowed token pieces) this matrix constrains
           generation in a way that model produces the right number of words, and so that the
           model (more or less) has to produce the words that are known to be correct.
  """
  # allowed_words[i] is words that are allowed after the ith space has been added
  allowed_words = []

  for i, const in enumerate(constraints):

    if isinstance(const, list):
      if i == 0:
        # Must start a new word at step 1
        allowed = starts_new_word.clone()
      elif isinstance(constraints[i - 1], str):
        # Either start any new word
        allowed = starts_new_word.clone()
        # Or continue the previous correct word
        allowed[tokenizer.encode(constraints[i - 1], add_special_tokens=False)[1:]] = 1
      else:
        # Anything goes, except special tokens
        allowed = torch.ones_like(starts_new_word, dtype=torch.float)
        allowed[tokenizer.all_special_ids] = 0
      allowed_words.append(allowed)

    else:
      input_id = tokenizer.encode(const, add_special_tokens=False)
      correct_word = input_id[0]
      if i == 0:
        # After the PAD token, can only start the new, correct word
        allowed = torch.zeros_like(starts_new_word)
        allowed[correct_word] = 1
      elif isinstance(constraints[i-1], str):
        allowed = torch.zeros_like(starts_new_word)
        # Anything that could continue the previous correct word
        allowed[tokenizer.encode_with_cache(constraints[i-1])[1:]] = 1
        allowed[correct_word] = 1  # And the start of the correct word
      else:
        allowed = continues_word.clone()  # Anything that does NOT start a new word
        allowed[correct_word] = 1  # And the start of the correct word
      allowed_words.append(allowed)

  allowed = continues_word.clone()  # Anything that does NOT start a new word
  allowed[tokenizer.eos_token_id] = 1 # And the EOS token
  allowed_words.append(allowed)
  return torch.stack(allowed_words, dim=0)


def t5_initialize_decoding(
    tokenizer, decoder, encoder_out, encoder_mask, constraints,
    constraints_before_softmax=False, post_process=None,
    prevent_known_incorrect_words=False
):
  batch_size = encoder_out.size(0)
  device = encoder_out.device
  starts_new_word = get_starts_word(tokenizer, decoder.config.vocab_size).to(device)

  continues_word = torch.logical_not(starts_new_word)
  continues_word[tokenizer.all_special_ids] = 0

  mat_size = max(len(x) for x in constraints) + 1
  allowed_word_mat = []
  for const in constraints:
    mat = build_allowed_word_matrix(const, starts_new_word, continues_word, tokenizer)
    mat = F.pad(mat, [0, 0, 0, mat_size - mat.size(0)])
    allowed_word_mat.append(mat)
  allowed_word_mat = torch.stack(allowed_word_mat, 0)
  allowed_word_mat = allowed_word_mat.to(torch.bool)

  initial_state = dict(
    encoder_mask=encoder_mask,
    batch_id=torch.arange(0, batch_size, device=device, dtype=torch.long),
    num_spaces=torch.zeros(batch_size, device=device, dtype=torch.long),
    encoder_outputs=encoder_out
  )

  banned_tensor = None
  if prevent_known_incorrect_words:
    banned_lists = [x for x in utils.flatten_list(constraints) if isinstance(x, list)]
    n_words = 0 if len(banned_lists) == 0 else max(len(x) for x in banned_lists)
    if n_words > 0:
      n_word_peices = max(len(tokenizer.encode(x, add_special_tokens=False)) for x in utils.flatten_list(banned_lists))
      banned_tensor = torch.full((len(constraints), mat_size, n_words, n_word_peices+1), -1, dtype=torch.long)
      banned_lens = torch.full((len(constraints), mat_size, n_words), -1, dtype=torch.long)
      for game_ix, const in enumerate(constraints):
        for word_ix, c in enumerate(const, start=1):
          if isinstance(c, list):
            for banned_ix, word in enumerate(c):
              word_pieces = tokenizer.encode(word, add_special_tokens=False)
              banned_lens[game_ix, word_ix, banned_ix] = len(word_pieces)
              for word_piece_ix, wp in enumerate(word_pieces):
                banned_tensor[game_ix, word_ix, banned_ix, word_piece_ix] = wp
      banned_tensor = banned_tensor.to(device=device).transpose(2, 3)
      banned_lens = banned_lens.to(device=device)
      banned_tensor = (banned_tensor, banned_lens)
      initial_state["num_word_pieces"] = torch.zeros(batch_size, device=device, dtype=torch.long)
      initial_state["on_banned"] = torch.zeros(batch_size, n_words, device=device, dtype=torch.bool)

  # extra tokens count as generating a space
  starts_new_word[32000:] = 1.0

  def _decode_step(predictions, prev_state):
    return decoding_step(decoder, predictions, prev_state, starts_new_word, continues_word,
                         allowed_word_mat, constraints_before_softmax, post_process, banned_tensor)

  initial_out = torch.full(
    (batch_size,), tokenizer.pad_token_id, dtype=torch.long, device=device)

  return initial_out, initial_state, _decode_step


def decoding_step(decoder, predictions, prev_state, starts_new_word, continues_word,
                  allowed_words_mat, constraints_before_softmax,
                  post_process=None, banned_tensor=None):
  num_spaces = prev_state["num_spaces"]
  batch_ids = prev_state["batch_id"]
  generated_space = starts_new_word[predictions]
  num_spaces = num_spaces + generated_space.to(torch.long)

  past = utils.flat_to_nested_struct({k: v.contiguous() for k, v in prev_state.items()
                                      if isinstance(k, tuple)})
  model_inputs = decoder.prepare_inputs_for_generation(
    predictions.unsqueeze(1), past=past, attention_mask=prev_state["encoder_mask"],
    encoder_outputs=(prev_state["encoder_outputs"],),
    use_cache=True)
  out: Seq2SeqLMOutput = decoder(**model_inputs, return_dict=True)

  next_state = dict(
    encoder_mask=prev_state["encoder_mask"],
    num_spaces=num_spaces,
    batch_id=batch_ids,
    encoder_outputs=prev_state["encoder_outputs"],
  )

  if post_process is not None:
    logits = post_process(out, model_inputs)
  else:
    logits = out.logits
  logits = logits.squeeze(1)
  cur_const = allowed_words_mat[batch_ids, num_spaces]

  if banned_tensor is not None:
    # Remove word pieces that would result in generating a banned (aka already-guessed) word
    # The high-level idea if we track what word number we are generating and what
    # word peice in that word we are on, then use that to figure out if we generating a
    # banned word for that word slot. If we are have completely generated a banned word,
    # we prevent the model from generating any word piece that would result in a space
    # so that generation complete that banned word.
    banned_tensor, banned_lens = banned_tensor
    on_piece = prev_state["num_word_pieces"]
    on_piece = torch.logical_not(generated_space).to(torch.long)*(1 + on_piece)
    next_state["num_word_pieces"] = on_piece

    on_banned = prev_state["on_banned"]   # [batch, n_words]
    on_banned[generated_space] = True
    batch, _, n_pieces, n_word = banned_tensor.size()
    on_piece = torch.min(on_piece, torch.as_tensor(n_pieces-1).to(on_piece))

    # [batch, n_words]
    continued_banned = banned_tensor[prev_state["batch_id"], num_spaces, on_piece]

    # [batch, n_words]
    continue_banned_lens = banned_lens[prev_state["batch_id"], num_spaces]

    on_banned = torch.logical_and(continued_banned == predictions.unsqueeze(1), on_banned)
    next_state["on_banned"] = on_banned
    is_banned_word = torch.any(torch.logical_and(on_banned, continue_banned_lens == (on_piece+1).unsqueeze(1)), 1)
    cur_const = cur_const.to(torch.bool)
    cur_const = torch.logical_and(cur_const, torch.logical_not(torch.logical_and(is_banned_word.unsqueeze(1), torch.logical_not(continues_word).unsqueeze(0))))

  cur_const = torch.logical_not(cur_const).float()

  if constraints_before_softmax:
    logits -= cur_const * 100000

  logits = F.log_softmax(logits, -1)

  if not constraints_before_softmax:
    logits -= cur_const * 100000

  utils.nested_struct_to_flat(out.past_key_values, (), cur_dict=next_state)
  return logits, next_state


def t5_decoding_token_constraints(tokenizer, decoder, encoder_out, encoder_mask,
                                  constraints, constraints_before_softmax=False):
  """
  Decode with per-token constraints, used for the drawer
  """
  batch_size = encoder_out.size(0)
  device = encoder_out.device

  initial_state = dict(
    encoder_mask=encoder_mask,
    encoder_outputs=encoder_out,
    time=torch.zeros(batch_size, device=encoder_out.device, dtype=torch.long)
  )

  def _decode_step(predictions, prev_state):
    return decoding_step_token_constraints(
      decoder, predictions, prev_state, constraints, constraints_before_softmax)

  initial_out = torch.full(
    (batch_size,), tokenizer.pad_token_id, dtype=torch.long, device=device)

  return initial_out, initial_state, _decode_step


def decoding_step_token_constraints(
    decoder, predictions, prev_state, constraints, constraints_before_softmax):
  time = prev_state["time"]

  past = utils.flat_to_nested_struct({k: v.contiguous() for k, v in prev_state.items()
                                      if isinstance(k, tuple)})
  model_inputs = decoder.prepare_inputs_for_generation(
    predictions.unsqueeze(1), past=past, attention_mask=prev_state["encoder_mask"],
    encoder_outputs=(prev_state["encoder_outputs"],),
    use_cache=True)
  out = decoder(**model_inputs, return_dict=True)
  next_state = dict(
    encoder_mask=prev_state["encoder_mask"],
    time=(time+1) % constraints.size(0),
    encoder_outputs=prev_state["encoder_outputs"]
  )
  logits = out.logits.squeeze(1)

  constraint = constraints[time[0]]
  if len(constraint.size()) == 2:
    constraint = constraint.unsqueeze(1)

  if constraints_before_softmax:
    logits -= (1 - constraint.float()) * 100000

  logits = F.log_softmax(logits, -1)

  if not constraints_before_softmax:
    logits -= (1 - constraint.float()) * 100000

  utils.nested_struct_to_flat(out.past_key_values, (), cur_dict=next_state)
  return logits, next_state


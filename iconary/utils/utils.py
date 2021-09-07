import json
import logging
import sys
from os import listdir, walk, makedirs
from os.path import exists, dirname, basename, join, relpath, isdir
from typing import Iterable, List, TypeVar, Tuple, Dict

import boto3
import torch
from allennlp.common.util import import_module_and_submodules


def cache_from_s3(bucket, file, out):
  if exists(out):
    return
  logging.info(f"Cached {bucket}/{file} to {out}")
  s3 = boto3.client('s3')
  if not exists(dirname(out)):
    makedirs(dirname(out))
  s3.download_file(bucket, file, out)


def import_all():
  """Import all modelling classes, used so we can load modules using FromParams"""
  import_module_and_submodules("iconary")


T = TypeVar('T')


def transpose_lists(lsts: Iterable[Iterable[T]]) -> List[List[T]]:
  """Transpose a list of lists."""
  return [list(i) for i in zip(*lsts)]


def to_device(batch, device):
  if batch is None:
    return None
  if isinstance(batch, (float, int, str)):
    return batch
  if isinstance(batch, dict):
    return {sub_k: to_device(sub_v, device) for sub_k, sub_v in batch.items()}
  if isinstance(batch, (tuple, list)):
    return [to_device(x, device) for x in batch]
  else:
    return batch.to(device)


def numpy(x):
  if x is None:
    return x
  return x.cpu().numpy()


def mask_logits(vec, mask):
  """Mask `vec` in logspace by setting out of bounds elements to very negative values"""
  if mask is None:
    return vec
  if vec.dtype == torch.float32:
    mask = mask.float()
    return vec * mask - (1 - mask) * 1E20
  else:
    raise NotImplementedError()


def _to_tuple(x):
  if isinstance(x, list):
    return tuple(_to_tuple(s) for s in x)
  return x


def load_json_object(file):
  with open(file, "r") as f:
    return json.load(f)


def dump_json_object(obj, file):
  with open(file, "w") as f:
    json.dump(obj, f)


def flatten_list(iterable_of_lists: Iterable[Iterable[T]]) -> List[T]:
  """Unpack lists into a single list."""
  return [x for sublist in iterable_of_lists for x in sublist]


def add_stdout_logger():
  handler = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                datefmt='%m/%d %H:%M:%S', )
  handler.setFormatter(formatter)
  handler.setLevel(logging.DEBUG)

  root = logging.getLogger()
  root.handlers = []  # Remove stderror handler that sometimes appears by default
  root.setLevel(logging.INFO)
  root.addHandler(handler)
  logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)


def is_model_dir(x):
  return exists(join(x, "model.json"))


def select_run_dir(run_dir):
  if exists(join(run_dir, "model.json")):
    candidates = []
    for filename in listdir(run_dir):
      filepath = join(run_dir, filename)
      if isdir(filepath) and filename.startswith("r"):
        candidates.append(filepath)
    if len(candidates) > 1:
      raise ValueError(f"Multiple runs in {run_dir}, please select one")
    elif len(candidates) == 0:
      raise ValueError(f"No runs found in {run_dir}")
    else:
      logging.info(f"Selecting run {basename(candidates[0])} for {run_dir}")
      return candidates[0]
  else:
    assert is_model_dir(basename(run_dir))
    return run_dir


def nested_struct_to_flat(tensors, prefix=(), cur_dict=None) -> Dict[Tuple, torch.Tensor]:
  if cur_dict is None:
    cur_dict = {}
    nested_struct_to_flat(tensors, (), cur_dict)
    return cur_dict

  if isinstance(tensors, torch.Tensor):
    cur_dict[prefix] = tensors
    return cur_dict

  if isinstance(tensors, dict):
    if len(tensors) == 0:
      raise ValueError("Cannot convert empty dict")
    for k, v in tensors.items():
      if isinstance(k, int):
        # TODO do we still need this?
        raise NotImplementedError("Integer keys")
      nested_struct_to_flat(v, prefix + (k, ), cur_dict)
  elif isinstance(tensors, (tuple, list)):
    if len(tensors) == 0:
      raise ValueError("Cannot convert empty tuples/lists")
    for ix, v in enumerate(tensors):
      nested_struct_to_flat(v, prefix + (ix, ), cur_dict)
  else:
    raise NotImplementedError()


def flat_to_nested_struct(nested: Dict[Tuple, torch.Tensor]):
  if len(nested) == 0:
    return None
  if isinstance(next(iter(nested.keys()))[0], str):
    out = {}
  else:
    out = []

  for prefix, value in nested.items():
    parent = out
    for i, key in enumerate(prefix[:-1]):
      next_parent = {} if isinstance(prefix[i+1], str) else []
      if isinstance(key, str):
        if key not in parent:
          parent[key] = next_parent
        parent = parent[key]

      elif isinstance(key, int):
        if len(parent) < key + 1:
          parent += [None] * (key + 1 - len(parent))
        if parent[key] is None:
          parent[key] = next_parent
        parent = parent[key]

      else:
        raise NotImplementedError()

    key = prefix[-1]
    if isinstance(key, int):
      if len(parent) < key + 1:
        parent += [None] * (key + 1 - len(parent))
    parent[prefix[-1]] = value

  return out
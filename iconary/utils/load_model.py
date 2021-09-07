import logging
from os import listdir
from os.path import dirname, join, exists

import torch
from allennlp.common import Registrable, Params
from iconary.models.iconary_model import IconaryModel
from iconary.utils import utils
from iconary.utils.utils import load_json_object


def load_model(run_dir, device=None, epoch=None) -> IconaryModel:
  """Loads an IconaryModel

  :param run_dir: Directory containing the model
  :param device: device to load the model to
  :param epoch: Specific epoch to load state from, otherwise load the latest one
  """
  utils.import_all()
  if run_dir.endswith("/"):
    run_dir = run_dir[:-1]
  model_spec = join(dirname(run_dir), "model.json")
  params = Params(load_json_object(model_spec))

  model: IconaryModel = IconaryModel.from_params(params)

  if epoch:
    src = join(run_dir, f"state-ep{epoch}.pth")
    if not exists(src):
      raise ValueError(f"Requested epoch {epoch} not found in {run_dir}")

  else:
    # Pick the latest epoch
    runs = [x for x in listdir(run_dir) if x.startswith("state-") and x.endswith(".pth")]
    if len(runs) == 0:
      src = join(run_dir, "state.pth")
      if not exists(src):
        raise ValueError(f"No states found in {run_dir}")
    else:
      src = join(run_dir, max(runs, key=lambda x: int(x.split(".")[0].split("-")[1][2:])))

  logging.info("Loading model state from %s" % src)
  # TODO is there way to efficently load the parameters straight to the gpu?
  state_dict = torch.load(src, map_location="cpu")
  model.load_state_dict(state_dict)
  if device is not None:
    model.to(device)
  model.eval()

  # Seems to help ensure state_dict to get freed from memory
  del state_dict

  return model

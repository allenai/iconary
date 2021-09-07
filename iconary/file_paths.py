from os import mkdir
from os.path import expanduser, dirname, join, exists

DATA_HOME = expanduser("~/data/iconary")
CACHE_HOME = join(dirname(dirname(__file__)), "data")
if not exists(CACHE_HOME):
  mkdir(CACHE_HOME)

ICONARY_HOME = join(DATA_HOME, "iconary")

TRAIN_VOC_CACHE = join(CACHE_HOME, "train_voc.json")

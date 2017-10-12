import pebl
import glob, os
from collections import namedtuple

DATA_PATH = "data"
THIS      = "datasets"
Model     = namedtuple("Model", [ "dataset", "prior" ])


def load(dataset):
  try:
    return getattr(__import__(THIS, fromlist=[dataset]), dataset)
  except:
    possible_data_files = glob.glob(DATA_PATH + "/*" + dataset + "*")

    if len(possible_data_files) > 1:
      raise Exception(
          "Too many matches for '%s': %s" %
          (dataset, possible_data_files))
    elif len(possible_data_files) == 1:
      return Model(pebl.data.fromfile(possible_data_files[0]), None)

  return None


def list():
  py_files = map(os.path.basename, glob.glob(THIS + "/*.py"))
  py_files.remove("__init__.py")
  return map(lambda x: x.split(".py")[0], py_files)

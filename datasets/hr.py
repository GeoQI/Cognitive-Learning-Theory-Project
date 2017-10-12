
DATA_PATH = "data/HR_tab_sep.pebl.tsv"

import numpy as np
from pebl import data
from pebl.prior import Prior

dataset = data.fromfile(DATA_PATH)
n_vars  = len(dataset.variables)

def prohibit_from_left(energy): return energy[6].sum() <= 0

for i in range(n_vars):
  if type(dataset.variables[i]) == data.ContinuousVariable:
    dataset.discretize(i, numbins=10)
  if type(dataset.variables[i]) == data.ClassVariable:
    dataset.discretize(i, numbins=dataset.variables[i].arity)

prohibited_edges = [ (6, i) for i in range(6) + range(6, 10) ]

prior = Prior(len(dataset.variables),
              #np.ones((n_vars, n_vars)) / (n_vars * n_vars))
              #constraints=[prohibit_from_left])
              prohibited_edges=prohibited_edges)


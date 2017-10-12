# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 16:04:37 2017

@author: Kathy
"""

from pebl import data, result
from pebl.learner import greedy, simanneal
import numpy as np
import sys
import shutil
import datasets

DEFAULT_DATASET     = datasets.load("greedytest")
DEFAULT_REPORT_DIR  = "./report/"

ds                  = DEFAULT_DATASET
report_dir          = DEFAULT_REPORT_DIR

if len(sys.argv) > 1:
  ds = datasets.load(sys.argv[1])
  if ds == None:
    raise Exception("Could not find dataset '%s'" % dataset)

if len(sys.argv) > 2:
  report_dir = sys.argv[2]


dataset = ds.dataset
prior   = ds.prior

greedy_lrn = greedy.GreedyLearner(dataset, prior, max_iterations=1000)
#anneal_lrn = simanneal.SimulatedAnnealingLearner(dataset, prior)
#results = result.merge(*[ greedy_lrn.run(), anneal_lrn.run() ])
results = greedy_lrn.run()

try:    shutil.rmtree(report_dir)
except: pass

results.tohtml(report_dir)


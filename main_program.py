# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 16:04:37 2017

@author: Kathy
"""

import read_dataset as data
import greedy_hill_climbing as greedy

dataset = data.fromfile("./testfiles/greedytest1-200.txt")
dataset.discretize()
learner = greedy.GreedyLearner(dataset)
ex1result = learner.run()
print"best network = " + str(ex1result.as_string())

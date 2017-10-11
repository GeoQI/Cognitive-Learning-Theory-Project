"""Learner that implements a greedy learning algorithm"""

import time
import copy
import network
import evaluator
from shared_utils import *
import numpy as N

class LearnerRunStats:
    def __init__(self, start):
        self.start = start
        self.end = None

class LearnerResult:
    """Class for storing any and all output of a learner.

    This is a mutable container for networks and scores. In the future, it will
    also be the place to collect statistics related to the learning task.

    """
    def __init__(self, learner_=None, size=None):
        self.data = learner_.data if learner_ else None
        self.nodes = self.data.variables if self.data else None
        self.size = size
        self.networks = []
        self.nethashes = {}
        self.best_network = []
        self.runs = []

    def start_run(self):
        """Indicates that the learner is starting a new run."""
        self.runs.append(LearnerRunStats(time.time()))

    def set_best_network(self, net):
        self.best_network = net
    
    def get_best_network(self):
        return self.best_network
    
    def stop_run(self):
        """Indicates that the learner is stopping a run."""
        self.runs[-1].end = time.time()


class CannotAlterNetworkException(Exception):
    pass

class Task(object):
    def run(self): pass

    def split(self, count):
        return [self] + [copy.deepcopy(self) for i in xrange(count-1)]
    
class Learner(Task):
    def __init__(self, data_=None, prior_=None, **kw):
        self.data = data_
        self.prior = prior_
        self.__dict__.update(kw)
        self.seed =''
        self.max_iterations=100
        # stats
        self.reverse = 0
        self.add = 0
        self.remove = 0

    def _alter_network_randomly_and_score(self):
        net = self.evaluator.network
        print "network = " + str(net.as_string())
        n_nodes = self.data.variables.size
        max_attempts = n_nodes**2

        # continue making changes and undoing them till we get an acyclic network
        for i in xrange(max_attempts):
            node1, node2 = N.random.random_integers(0, n_nodes-1, 2)    
            print "node1=" + str(node1) + " ,node2=" + str(node2)
            if (node1, node2) in net.edges:
                # node1 -> node2 exists, so reverse it.    
                add,remove = [(node2, node1)], [(node1, node2)]
                print "switching node1 and node2"
            elif (node2, node1) in net.edges:
                # node2 -> node1 exists, so remove it
                add,remove = [], [(node2, node1)]
                print "removing node1 from node2"
            else:
                # node1 and node2 unconnected, so connect them
                add,remove =  [(node1, node2)], []
                print "adding node1 and node2"
            try:
                score = self.evaluator.alter_network(add=add, remove=remove)
            except evaluator.CyclicNetworkError:
                continue # let's try again!
            else:
                if add and remove:
                    self.reverse += 1
                elif add:
                    self.add += 1
                else:
                    self.remove += 1
                return score

        # Could not find a valid network  
        raise CannotAlterNetworkException() 

class LearnerStatistics:
    def __init__(self):
        self.restarts = -1
        self.iterations = 0
        self.unimproved_iterations = 0
        self.best_score = 0
        self.start_time = time.time()

class GreedyLearner(Learner):

    def __init__(self, data_=None, prior_=None, **options):
        """
        Create a learner that uses a greedy learning algorithm.

        The algorithm works as follows:

            1. start with a random network
            2. Make a small, local change and rescore network
            3. If new network scores better, accept it, otherwise reject.
            4. Steps 2-3 are repeated till the restarting_criteria is met, at
               which point we begin again with a new random network (step 1)
                    
        """

        super(GreedyLearner, self).__init__(data_, prior_)
        self.options = options
        if not isinstance(self.seed, network.Network):
            self.seed = network.Network(self.data.variables, self.seed)
        
    def run(self):
        """Run the learner.

        Returns a LearnerResult instance. Also sets self.result to that
        instance.  
        
        """

        # max_time and max_iterations are mutually exclusive stopping critera
        _stop = self._stop_after_iterations
        self.stats = LearnerStatistics()
        self.result = LearnerResult(self)
        self.evaluator = evaluator.SmartNetworkEvaluator(self.data, self.seed, self.prior)
        self.evaluator.score_network(self.seed.copy())
        first = True
        self.result.start_run()
        while not _stop():
            self._run_without_restarts(_stop)
            first = False
        self.result.stop_run()
        print "best score " + str(self.stats.best_score)
        return self.result.get_best_network()

    def _run_without_restarts(self, _stop):
        self.stats.restarts += 1
        self.stats.unimproved_iterations = 0
         
        # set the default best score
        self.stats.best_score = self.evaluator.score_network()

        # continue learning until time to stop or restart
        while not (_stop()):
            self.stats.iterations += 1
            print "iteration = " + str(self.stats.iterations)
            try:
                curscore = self._alter_network_randomly_and_score()
                print "score =" + str(curscore)
            except CannotAlterNetworkException:
                return
            

            if curscore <= self.stats.best_score:
                # score did not improve, undo network alteration
                self.stats.unimproved_iterations += 1
                self.evaluator.restore_network()
            else:
                self.result.set_best_network(self.evaluator.network)
                self.stats.best_score = curscore
                self.stats.unimproved_iterations = 0

    #
    # Stopping and restarting criteria
    # 

    def _stop_after_iterations(self):
        return self.stats.iterations >= self.max_iterations

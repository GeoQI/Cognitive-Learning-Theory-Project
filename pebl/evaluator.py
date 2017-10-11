"""Classes and functions for efficiently evaluating networks."""

import numpy as N
from collections import deque
import cpd
import prior
from shared_utils import *

N.random.seed()

#
# Exceptions
#
class CyclicNetworkError(Exception):
    msg = "Network has cycle and is thus not a DAG."


#
# Localscore Cache
#
class LocalscoreCache(object):
    """ A LRU cache for local scores.

    Based on code from http://code.activestate.com/recipes/498245/
    """

    def __init__(self, evaluator, cachesize=None):
        self._cache = {}
        self._queue = deque()
        self._refcount = {}
        self.cachesize = cachesize

        self.neteval = evaluator
        self.hits = 0
        self.misses = 0

    def __call__(self, node, parents):
        # make variables local
        _len = len
        _queue = self._queue
        _refcount = self._refcount
        _cache = self._cache
        _maxsize = self.cachesize

        index = tuple([node] +  parents)
        
        # get from cache or compute
        try:
            score = _cache[index]
            self.hits += 1
        except KeyError:
            score = _cache[index] = self.neteval._cpd(node, parents).loglikelihood()
            self.misses += 1

        # if using LRU cache (maxsize != -1)
        if _maxsize > 0:
            # record that key was accessed
            _queue.append(index)
            _refcount[index] = _refcount.get(index, 0) + 1

            # purge LRU entry
            while _len(_cache) > _maxsize:
                k = _queue.popleft()
                _refcount[k] -= 1
                if not _refcount[k]:
                    del _cache[k]
                    del _refcount[k]

            # Periodically compact the queue by duplicate keys
            if _len(_queue) > _maxsize * 4:
                for i in xrange(_len(_queue)):
                    k = _queue.popleft()
                    if _refcount[k] == 1:
                        _queue.append(k)
                    else:
                        _refcount[k] -= 1
            
        return score



#
# Network Evaluators
#
class NetworkEvaluator(object):
    """Base Class for all Network Evaluators.

    Provides methods for scoring networks but does not eliminate any redundant
    computation or cache retrievals.
    
    """

    def __init__(self, data_, network_, prior_=None, localscore_cache=None):

        self.network = network_
        self.data = data_
        self.prior = prior_ or prior.NullPrior()
        
        self.datavars = range(self.data.variables.size)
        self.score = None
        self._localscore = localscore_cache or LocalscoreCache(self)
        self.localscore_cache = self._localscore

    #
    # Private Interface
    # 
    def _globalscore(self, localscores):
        # log(P(M|D)) +  log(P(M)) == likelihood + prior
        return N.sum(localscores) + self.prior.loglikelihood(self.network)
    
    def _cpd(self, node, parents):
        return cpd.MultinomialCPD(
            self.data._subset_ni_fast([node] + parents))


    def _score_network_core(self):
        # in this implementation, we score all nodes (even if that means
        # redundant computation)
        parents = self.network.edges.parents
        self.score = self._globalscore(
            self._localscore(n, parents(n)) for n in self.datavars
        )
        return self.score

    #
    # Public Interface
    #
    def score_network(self, net=None):
        """Score a network.

        If net is provided, scores that. Otherwise, score network previously
        set.

        """

        self.network = net or self.network
        return self._score_network_core()

    def alter_network(self, add=[], remove=[]):
        """Alter network by adding and removing sets of edges."""

        self.network.edges.add_many(add)
        self.network.edges.remove_many(remove)
        return self.score_network()
    
    def clear_network(self):     
        """Clear all edges from the network."""

        self.network.edges.clear()
        return self.score_network()


class SmartNetworkEvaluator(NetworkEvaluator):
    def __init__(self, data_, network_, prior_=None, localscore_cache=None):
        """Create a 'smart' network evaluator.

        This network evaluator eliminates redundant computation by keeping
        track of changes to network and only rescoring the changes. This
        requires that all changes to the network are done through this
        evaluator's methods. 

        The network can be altered by the following methods:
            * alter_network
            * score_network
            * randomize_network
            * clear_network

        The last change applied can be 'undone' with restore_network

        """

        super(SmartNetworkEvaluator, self).__init__(data_, network_, prior_, 
                                                    localscore_cache)


        # these represent that state that we intelligently manage
        self.localscores = N.zeros((self.data.variables.size), dtype=float)
        self.dirtynodes = set(self.datavars)
        self.saved_state = None

    #
    # Private Interface
    #
    def _backup_state(self, added, removed):
        self.saved_state = (
            self.score,                     # saved score
            #[(n,self.localscores[n]) for n in self.dirtynodes],
            self.localscores.copy(),        # saved localscores
            added,                          # edges added
            removed                         # edges removed
        )

    def _restore_state(self):
        if self.saved_state:
            self.score, self.localscores, added, removed = self.saved_state
            #self.score, changedscores, added, removed = self.saved_state
        
        #for n,score in changedscores:
            #self.localscores[n] = score

        self.network.edges.add_many(removed)
        self.network.edges.remove_many(added)
        self.saved_state = None
        self.dirtynodes = set()

    def _score_network_core(self):
        # if no nodes are dirty, just return last score.
        if len(self.dirtynodes) == 0:
            print"nothing changed, self.score = " + str(self.score)
            return self.score
        
        # update localscore for dirtynodes, then re-calculate globalscore
        parents = self.network.edges.parents
        for node in self.dirtynodes:
            self.localscores[node] = self._localscore(node, parents(node))
        
        self.dirtynodes = set()
        self.score = self._globalscore(self.localscores)

        return self.score
    
    #
    # Public Interface
    #
    def score_network(self, net=None):
        print "scoring network!"
        """Score a network.

        If net is provided, scores that. Otherwise, score network previously
        set.

        """

        if net:
            add = [edge for edge in net.edges if edge not in self.network.edges]
            remove = [edge for edge in self.network.edges if edge not in net.edges]
        else:
            add = remove = []
        return self.alter_network(add, remove)
    
    def alter_network(self, add=[], remove=[]):
        """Alter the network while retaining the ability to *quickly* undo the changes."""

        # make the required changes
        # NOTE: remove existing edges *before* adding new ones. 
        #   if edge e is in `add`, `remove` and `self.network`, 
        #   it should exist in the new network. (the add and remove cancel out.
        self.network.edges.remove_many(remove)
        self.network.edges.add_many(add)    

        # check whether changes lead to valid DAG (raise error if they don't)
        affected_nodes = set(unzip(add, 1))
        if affected_nodes and not self.network.is_acyclic(affected_nodes):
            self.network.edges.remove_many(add)
            self.network.edges.add_many(remove)
            raise CyclicNetworkError()
        
        
        # accept changes: 
        #   1) determine dirtynodes
        #   2) backup state
        #   3) score network (but only rescore dirtynodes)
        self.dirtynodes.update(set(unzip(add+remove, 1)))
        self._backup_state(add, remove)
        self.score = self._score_network_core()
        #print"calculated score = " + str(self.score)
        return self.score


    def clear_network(self):
        """Clear all edges from the network."""

        return self.alter_network(remove=list(self.network.edges))

    def restore_network(self):
        """Undo the last change to the network (and score).
        
        Undo the last change performed by any of these methods:
            * score_network
            * alter_network
            * randomize_network
            * clear_network

        """

        self._restore_state()
        return self.score

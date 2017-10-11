"""Classes for representing networks and functions to create/modify them."""

from copy import  deepcopy
from bisect import insort
import numpy as N
from shared_utils import *

try:
    from network import _network
except:
    _network = None

class EdgeSet(object):
    """
    Maintains a set of edges.

    Performance characteristics:
        - Edge insertion: O(1)
        - Edge retrieval: O(n)
    
    Uses adjacency lists but exposes an adjacency matrix interface via the
    adjacency_matrix property.

    """

    def __init__(self, num_nodes=0):
        self._outgoing = [[] for i in xrange(num_nodes)]
        self._incoming = [[] for i in xrange(num_nodes)] 
        
    def clear(self):
        """Clear the list of edges."""
        self.__init__(len(self._outgoing)) 

    def add(self, edge):
        """Add an edge to the list."""
        self.add_many([edge])
        
    def add_many(self, edges):
        """Add multiple edges."""

        for src,dest in edges:
            if dest not in self._outgoing[src]: 
                insort(self._outgoing[src], dest)
                insort(self._incoming[dest], src)
            
    def remove(self, edge):
        """Remove edges from edgelist.
        
        If an edge to be removed does not exist, fail silently (no exceptions).

        """
        self.remove_many([edge])

    def remove_many(self, edges):
        """Remove multiple edges."""

        for src,dest in edges:
            try: 
                self._incoming[dest].remove(src)
                self._outgoing[src].remove(dest)
            except KeyError, ValueError: 
                pass

    def incoming(self, node):
        """Return list of nodes (as node indices) that have an edge to given node.
        
        The returned list is sorted.
        Method is also aliased as parents().
        
        """
        return self._incoming[node]

    def outgoing(self, node):
        """Return list of nodes (as node indices) that have an edge from given node.
        
        The returned list is sorted.
        Method is also aliased as children().

        """
        return self._outgoing[node]

    parents = incoming
    children = outgoing

    def __iter__(self):
        """Iterate through the edges in this edgelist.

        Sample usage:
        for edge in edgelist: 
            print edge

        """
        
        for src, dests in enumerate(self._outgoing):
            for dest in dests:
                yield (src, dest)

    def __eq__(self, other):
        for out1,out2 in zip(self._outgoing, other._outgoing):
            if out1 != out2:
                return False
        return True

    def __hash__(self):
        return hash(tuple(tuple(s) for s in self._outgoing))
        
    def __copy__(self):
        other = EdgeSet.__new__(EdgeSet)
        other._outgoing = [[i for i in lst] for lst in self._outgoing]
        other._incoming = [[i for i in lst] for lst in self._incoming]
        return other

    def as_tuple(self):
        return tuple(tuple(s) for s in self._outgoing)
    
    @extended_property
    def adjacency_matrix():
        """Set or get edges as an adjacency matrix.

        The adjacency matrix is a boolean numpy.ndarray instance.

        """

        def fget(self):
            size = len(self._outgoing)
            adjmat = N.zeros((size, size), dtype=bool)
            selfedges = list(self)
            if selfedges:
                adjmat[unzip(selfedges)] = True
            return adjmat

        def fset(self, adjmat):
            self.clear()
            for edge in zip(*N.nonzero(adjmat)):
                self.add(edge)

        return locals()

    @extended_property
    def adjacency_lists():
        """Set or get edges as two adjacency lists.

        Property returns/accepts two adjacency lists for outgoing and incoming
        edges respectively. Each adjacency list if a list of sets.

        """

        def fget(self):
            return self._outgoing, self._incoming

        def fset(self, adjlists):
            if len(adjlists) is not 2:
                raise Exception("Specify both outgoing and incoming lists.")
           
            # adjlists could be any iterable. convert to list of lists
            _outgoing, _incoming = adjlists
            self._outgoing = [list(lst) for lst in _outgoing]
            self._incoming = [list(lst) for lst in _incoming]

        return locals()


class Network(object):
    """A network is a set of nodes and directed edges between nodes"""
    

    #
    # Public methods
    #
    def __init__(self, nodes, edges=None):
        """Creates a Network.

        nodes is a list of pebl.data.Variable instances.
        edges can be:

            * an EdgeSet instance
            * a list of edge tuples
            * an adjacency matrix (as boolean numpy.ndarray instance)
            * string representation (see Network.as_string() for format)

        """
        
        self.nodes = nodes
        self.nodeids = range(len(nodes))

        # add edges
        if isinstance(edges, EdgeSet):
            self.edges = edges
        elif isinstance(edges, N.ndarray):
            self.edges = EdgeSet(len(edges))
            self.edges.adjacency_matrix = edges    
        else:
            self.edges = EdgeSet(len(self.nodes))

    def is_acyclic(self, roots=None):
        """Uses a depth-first search (dfs) to detect cycles."""

        roots = list(roots) if roots else self.nodeids
        if _network:
            return _network.is_acyclic(self.edges._outgoing, roots, [])
        else:
            return self.is_acyclic_python(roots)


        #---------------

        children = self.edges.children
        roots = set(roots) if roots else set(range(len(self.nodes)))
        return _isacyclic(roots, set())
    
    def is_acyclic_python(self, roots=None):
        """Uses a depth-first search (dfs) to detect cycles."""

        def _isacyclic(tovisit, visited):
            if tovisit.intersection(visited):
                # already visited a node we need to visit. thus, cycle!
                return False

            for n in tovisit:
                # check children for cycles
                if not _isacyclic(set(children(n)), visited.union([n])):
                    return False

            # got here without returning false, so no cycles below rootnodes
            return True

        #---------------

        children = self.edges.children
        roots = set(roots) if roots else set(range(len(self.nodes)))
        return _isacyclic(roots, set())

#    # TODO: test
    def copy(self):
        """Returns a copy of this network."""
        newedges = EdgeSet(len(self.nodes))
        newedges.adjacency_lists = deepcopy(self.edges.adjacency_lists)

        return Network(self.nodes, newedges)    
      
    def as_string(self):
        """Returns the sparse string representation of network.

        If network has edges (2,3) and (1,2), the sparse string representation
        is "2,3;1,2".

        """

        return ";".join([",".join([str(n) for n in edge]) for edge in list(self.edges)])
     
    def as_dotstring(self):
        """Returns network as a dot-formatted string"""

        def node(n, position):
            s = "\t\"%s\"" % n.name
            if position:
                x,y = position
                s += " [pos=\"%d,%d\"]" % (x,y)
            return s + ";"


        nodes = self.nodes
        positions = self.node_positions if hasattr(self, 'node_positions') \
                                        else [None for n in nodes]

        return "\n".join(
            ["digraph G {"] + 
            [node(n, pos) for n,pos in zip(nodes, positions)] + 
            ["\t\"%s\" -> \"%s\";" % (nodes[src].name, nodes[dest].name) 
                for src,dest in self.edges] +
            ["}"]
        )


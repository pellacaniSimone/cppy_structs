
"""

Author: Simone Pellacani (pellacanisimone2017@gmail.com)
Brief: This module provides graph data structures and algorithms.
Version: 0.0.1
Date: 2025-05-10

Copyright (c) 2025

Refactored and documented using LLMs

This module provides a foundational graph library with support for directed and undirected graphs,
including specialized Directed Acyclic Graphs (DAGs). It features a mixin pattern implementation
with `GraphMixin` as the base class containing core graph functionality, which is then inherited
by concrete graph implementations.

The `GraphMixin` class serves as the core implementation, providing functionalities for:
- Vertex and edge management (addition/removal)
- Graph density calculation
- Graph transposition
- Matrix representation conversion
- String representation of graphs

The `Graph` class inherits from `GraphMixin` and acts as a general-purpose graph implementation
supporting both directed and undirected edges with optional weights and tags.

The `DAG` class is a specialized subclass of `GraphMixin` for Directed Acyclic Graphs, featuring
cycle detection through depth-first search.

Key Features:
    - Support for both directed and undirected edges
    - Vertex and edge tagging system for metadata storage
    - Automatic density calculation (sparse/dense graphs)
    - Graph transposition operation
    - Conversion to adjacency matrix representation
    - Cycle detection for DAG validation
    - Clean string representation of graph structure

Graph Representation:
    - Vertices: Stored as dictionary keys with optional tags
    - Edges: Stored as dictionary of tuples (source, target) with weight and tag lists
    - Density: Calculated as edges/vertices ratio (0=empty, >0=dense, <0=sparse)

Usage:
    - Import the desired class (`Graph` or `DAG`) from the module
    - Create instances and add vertices/edges
    - Perform graph operations and validations
    - Convert to matrix representation when needed

Example:
    >>> graph = Graph()
    >>> graph.add_vertex(0, 'start')
    >>> graph.add_vertex(1, 'middle')
    >>> graph.add_edge(0, 1, 2.5, 'connection')
    >>> print(graph)
    >>> matrix = graph.to_matrix()

    >>> dag = DAG()
    >>> dag.add_vertex(0)
    >>> dag.add_vertex(1)
    >>> dag.add_arc(0, 1, 1.0, 'dependency')
    >>> print(dag.is_dag())  # Should return True

"""





class GraphMixin:
    # used here results in shared mem var
    def __init__(self):
        self.dim : int = 0 ; self.density : float = 0; self.edges_tags : int = 0
        self.vertex : dict = {} # {w : ['TAG']}
        self.edges : dict ={} # {(u,v) : [ weigh, ['TAG'] ]}
    """
        Density:
        - d==0 -> empty
        - d>1  -> dense
        - d<1  -> sparse
    """
    def graph_density(self):self.density = len(self.edges)/ len(self.vertex) if self.vertex else 0
    def increase_size(self):self.dim+=1;self.graph_density()
    def decrease_size(self):self.dim-=1;self.graph_density()
    def add_vertex(self, w: int, tag: str = ''):
        if w not in self.vertex: self.vertex[w] = tag ; self.increase_size()
    def add_arc(self, u: int, v: int, w: float, tag:str) : 
        if (u, v) not in self.edges: self.edges[(u, v)] = []
        if (w, tag) not in self.edges[(u, v)]: self.edges[(u, v)].append((w, tag))
        self.graph_density()
    def add_edge(self, u: int, v: int, w: float, tag:str ) : self.add_arc( u, v, w, tag) ; self.add_arc( v, u, w, tag)
    def add(self, u: int, v: int, w: float=1.0,  tag:str='' ,directed: bool=True): 
        if directed:  self.add_arc( u, v, w,tag)
        else: self.add_edge( u, v, w,tag)
    def del_edge(self, u,v, directed: bool=True): 
        del self.edges[(u,v)]
        if not directed: del self.edges[(v,u)] ; self.graph_density()
    def del_vertex(self, key: int): # and all its edges
        del self.vertex[key];self.decrease_size()
        for k in self.edges:  self.del_edge(k) if key == k else ...
    def t(self):
        Gt = self.__class__()  # same current class
        Gt.vertex = self.vertex.copy()
        Gt.dim = self.dim
        Gt.edges = {(v, u): k.copy() for (u, v), k in self.edges.items()}
        return Gt
    def __str__(self) -> str:
        out = f"Graph - Size: {self.dim}, Density: {self.density:.2f}\n"
        out += "Vertices:\n"
        for k, tag in self.vertex.items():
            if tag != '': out += f"{tag}"
            else: out += f"{k}"
            for (u, v), edlis in self.edges.items():
                if u == k:
                    term=self.vertex[v] if self.vertex[v] else v
                    if edlis != [] : out += f" --({edlis})--> {term}"
                    else :  out += f" --> {term}"
            out += "\n"
        return out
    def to_matrix(self, weighted=True) :
        """Return square matrix"""
        from algebra import Matrix
        size = len(self.vertex)
        mat = Matrix((size, size), command='zeros')
        for k, _ in self.vertex.items():
            for (u, v), edlis in self.edges.items():
                if u == k:
                    if weighted:
                        mat[k][v]=edlis[0]
                    else:
                        mat[k][v]=1
        return mat


class Graph(GraphMixin):
    pass

class DAG(GraphMixin):
    """Directed Acyclic Graph"""
    def is_dag(self) -> bool:
        visited = set()
        rec_stack = set()

        def dfs(v):
            visited.add(v)
            rec_stack.add(v)
            for (u, w), _ in self.edges.items():
                if u == v:
                    if w not in visited:
                        if dfs(w):
                            return True
                    elif w in rec_stack:
                        return True
            rec_stack.remove(v)
            return False

        for node in self.vertex:
            if node not in visited:
                if dfs(node):
                    return False
        return True
  

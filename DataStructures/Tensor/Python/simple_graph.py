
class GraphMixin:
    # used here results in shared mem var
    def __init__(self):
        self.dim : int = 0 ; self.density : float = 0; self.edges_tags : int = 0
        self.vertex : dict = {} # {w : ['TAG']}
        self.edges : dict ={} # {(u,v) : [ weigh, ['TAG'] ]}
    """
        Density:
        - d==0 -> empty
        - d>0  -> dense
        - d<0  -> sparse
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
  

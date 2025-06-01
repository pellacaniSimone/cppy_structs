from array import array
import random, os, math
from typing import List, Tuple, Union, Set, Optional, Dict
from collections import deque


class GenericTensor(array):
    """Base class for tensor operations"""
    
    __allowed_cmds = {'zeros', 'ones', 'rand', 'randn', 'randi', 'triu', 'tril', 'diag', 'eye', 'sprand', 'inf'}

    @staticmethod
    def _fill_triangle(data: List[float], shape: tuple|int, col_range) -> List[float]:
        """Fill triangular portion of matrix with data"""
        r, c = shape
        mat = [0.0] * math.prod(shape)
        for i in range(r):
            for j in col_range(i):
                idx = i * c + j
                if idx < len(data):
                    mat[idx] = data[idx]
        return mat

    @staticmethod
    def _empty(shape: tuple|int) -> List[float]:
        """Create empty tensor with given shape"""
        return [0.0] * math.prod(shape)

    @staticmethod
    def _diag_matrix(shape: tuple|int, vals: List[float]) -> List[float]:
        """Create diagonal matrix with given values"""
        mat = GenericTensor._empty(shape)
        r = shape[0]
        for i in range(min(r, len(vals))):
            mat[i * (r + 1)] = vals[i]
        return mat

    @staticmethod
    def _sparse_matrix(shape: tuple|int, density: float) -> List[float]:
        """Create sparse matrix with given density"""
        n = math.prod(shape)
        mat = GenericTensor._empty(shape)
        cnt = int(n * density)
        for pos in random.sample(range(n), cnt):
            mat[pos] = random.uniform(0, 1)
        return mat

    def __new__(cls, shp: tuple|int, cmd: str = 'zeros', data: list = None, **kwargs):
        """Create new tensor instance"""
        shp = tuple(shp)
        n = 1
        l = kwargs.get('min_val', 0)
        r = kwargs.get('max_val', 10)
        for d in shp:
            if d < 1 or (len(shp)==1 and d == 1 ) or isinstance(d, float) :
                raise ValueError("Shape dimensions must contain positive integers")
            n *= d
        if data is not None:
            if len(data) != n:
                raise ValueError("Data length does not match the product of the shape dimensions")
            init = data
        else:
            if cmd not in cls.__allowed_cmds:
                raise ValueError(f"Invalid command. Allowed values: {cls.__allowed_cmds}")
            if cmd == 'zeros':
                init = [0.0] * n
            elif cmd == 'ones':
                init = [1.0] * n
            elif cmd == 'rand':
                init = [random.uniform(l, r) for _ in range(n)]
            elif cmd == 'randn':
                init = [random.gauss(l, r) for _ in range(n)]
            elif cmd == 'inf':
                init = [float('inf')] * n
            elif cmd == 'triu':
                return GenericTensor._fill_triangle(kwargs['data'], shp, lambda i: range(i, shp[1]))
            elif cmd == 'tril':
                return GenericTensor._fill_triangle(kwargs['data'], shp, lambda i: range(i+1))
            elif cmd == 'diag':
                return GenericTensor._diag_matrix(shp, kwargs['data'])
            elif cmd == 'eye':
                return GenericTensor._diag_matrix(shp, [1.0]*shp[0])
            elif cmd == 'sprand':
                return GenericTensor._sparse_matrix(shp, kwargs.get('density', 0.1))

        obj = super().__new__(cls, 'd', init)
        obj.shape = tuple((1,)) if n == 1 else shp
        obj.size = n
        s = [1]
        for d in reversed(obj.shape[1:]): s.insert(0, s[0] * d)
        obj._strides = tuple(s)
        return obj
    


    def _flatten_idx(self, idx: Tuple[int, ...]) -> int:
        """Convert multi-dimensional index to flat index"""
        if len(idx) != len(self.shape):
            raise IndexError("Incorrect number of indices")
        return sum(i * s for i, s in zip(idx, self._strides))

    def __getitem__(self, idx):
        """Get tensor element at given index"""
        if isinstance(idx, int):
            idx = (idx,)
        if len(idx) < len(self.shape):
            sub_shape = self.shape[len(idx):]
            base = self._flatten_idx(idx + (0,) * (len(self.shape) - len(idx)))
            sub_n = 1
            for d in sub_shape:
                sub_n *= d
            sub_dat = [array.__getitem__(self, base + i) for i in range(sub_n)]
            return type(self)(sub_shape, data=sub_dat)
        return array.__getitem__(self, self._flatten_idx(idx))

    def __setitem__(self, idx, val):
        """Set tensor element at given index"""
        if isinstance(idx, int):
            idx = (idx,)
        if len(idx) != len(self.shape):
            raise IndexError("Incorrect number of indices")
        flat_idx = self._flatten_idx(idx)
        super().__setitem__(flat_idx, val)

    def __repr__(self):
        """String representation of tensor"""
        return f"{self.__class__.__name__}(shape={self.shape}, data={list(self)})"

    def __str__(self) -> str:
        """Pretty string representation of tensor"""
        def build_string(indices: list, depth: int) -> str:
            if depth == len(self.shape):
                return f"{self[tuple(indices)]:.2f}"
            parts = []
            for i in range(self.shape[depth]):
                new_indices = indices + [i]
                part = build_string(new_indices, depth + 1)
                parts.append(part)
            return "[ " + ", ".join(parts) + " ]"
        return build_string([], 0)

    def save(self, filename: str) -> None:
        """Save tensor to file"""
        with open(filename, 'w') as f:
            f.write(f"{len(self.shape)}\n")
            f.write(" ".join(map(str, self.shape)) + "\n")
            f.write(" ".join(f"{val:.6f}" for val in self) + "\n")

    @classmethod
    def load(cls, filename: str):
        """Load tensor from file"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist")
        with open(filename, 'r') as f:
            num_dims = int(f.readline().strip())
            shape = tuple(map(int, f.readline().strip().split()))
            if len(shape) != num_dims:
                raise ValueError("Mismatch in the number of dimensions")
            data = list(map(float, f.readline().strip().split()))
            return cls(shape, data=data)

    @classmethod
    def __reshape__(cls, T: 'GenericTensor') -> 'GenericTensor':
        """Reshape tensor"""
        if T.size == 1:
            s = (1, T.shape[0]) if T is cls else (T.shape[0], 1)
            return type(GenericTensor)(s, data=list(T))
        return T

    def __len__(self):
        """Get total number of elements"""
        n = 1
        for d in self.shape:
            n *= d
        return n

    def __proxy_op(self, B, op):
        if self.shape != B.shape:
            raise ValueError("Shapes must match for operation")
        return type(self)(self.shape, data=[op(a, b) for a, b in zip(self, B)])

    def __add__(self, B):
        return self.__proxy_op(B, lambda a, b: a + b)

    def __sub__(self, B):
        return self.__proxy_op(B, lambda a, b: a - b)

    def __neg__(self):
        return type(self)(self.shape, data=[-a for a in self])

    def __div__(self, B):
        return self.__proxy_op(B, lambda a, b: a / b if b else None)

    def __mul__(self, B):
        """Multiply tensors elementwise or by scalar"""
        if isinstance(B, GenericTensor):
            return self.__proxy_op(B, lambda a, b: a * b)
        else:
            return type(self)(self.shape, data=[a * B for a in self])

    def _matmul_core(self, B: "GenericTensor") -> Tuple[List[float], Tuple[int, ...]]:
        """Core matrix multiplication implementation"""
        A_shape, B_shape = self.shape, B.shape
        A_view, B_view = self.__reshape__(self), self.__reshape__(B)
        if A_view.shape[-1] != B_view.shape[0]:
            raise ValueError(f"Shapes not aligned for matrix multiplication: {A_shape} and {B_shape}")
        out_shape = A_view.shape[:-1] + B_view.shape[1:]
        code_lines, indent, total_loops = [], "", len(out_shape)
        for idx, dim in enumerate(out_shape):
            var = f"i{idx}"
            code_lines.append(f"{indent}for {var} in range({dim}):")
            indent += "    "
        code_lines.append(f"{indent}s = 0.0")
        k_dim = A_view.shape[-1]
        code_lines.append(f"{indent}for k in range({k_dim}):")
        a_idx = [f"i{i}" for i in range(len(A_view.shape)-1)] + ["k"]
        b_idx = ["k"] + [f"i{i}" for i in range(len(A_view.shape)-1, total_loops)]
        code_lines.append(f"{indent}    s += A_view[{', '.join(a_idx)}] * B_view[{', '.join(b_idx)}]")
        code_lines.append(f"{indent}result.append(s)")
        exec_code = "\n".join(code_lines)
        local_vars = {"A_view": A_view, "B_view": B_view, "result": []}
        exec(exec_code, {}, local_vars)
        return local_vars["result"], out_shape

    def __matmul__(self, B: "GenericTensor") -> "GenericTensor":
        result, out_shape = self._matmul_core(B)
        if len(out_shape) == 1:  
            return Vector(out_shape[0], data=result)
        elif len(out_shape) == 2: 
            return Matrix(out_shape, data=result)
        else: 
            return Tensor(out_shape, data=result)

    @staticmethod
    def machine_precision():
        """Calculate machine precision"""
        eps = 1.0
        while 1.0 + eps != 1.0:
            eps_last = eps
            eps /= 2.0
        return eps_last

class Tensor(GenericTensor):
    def __new__(cls, shape: tuple|int, command: str = 'zeros', data: List[float] = None) -> "Tensor":
        return super().__new__(cls, shape, command, data)
        
    def __add__(self, B) -> 'Tensor':
        result = super().__add__(B)
        return Tensor(result.shape, data=list(result))

    def __mul__(self, B: Union[float, int, "Tensor"]) -> 'Tensor':
        result = super().__mul__(B)
        return Tensor(result.shape, data=list(result))

    @classmethod
    def __reshape__(cls, T: "Tensor" = None) -> 'Tensor':
        result = super().__reshape__(T)
        return Tensor(result.shape, data=list(result))

    def __matmul__(self, B: "Tensor") -> 'Tensor':
        result, out_shape = self._matmul_core(B)
        return Tensor(out_shape, data=result)

class Matrix(GenericTensor):
    def __new__(cls, shape, command: str = 'zeros', data: List[float] = None) -> "Matrix":
        return super().__new__(cls, shape, command, data)



    def __add__(self, B) -> 'Matrix':
        result = super().__add__(B)
        return Matrix(result.shape, data=list(result))

    def __mul__(self, B: Union[float, int, "Matrix"]) -> 'Matrix':
        result = super().__mul__(B)
        return Matrix(result.shape, data=list(result))

    @classmethod
    def __reshape__(cls, T: "Matrix" = None) -> 'Matrix':
        result = super().__reshape__(T)
        return Matrix(result.shape, data=list(result))

    def __matmul__(self, B: "Matrix") -> 'Matrix':
        result, out_shape = self._matmul_core(B)
        return Matrix(out_shape, data=result)



    def t(self) -> "Matrix":
        """Transpose"""
        r, c = self.shape
        data_t = [self[i, j] for j in range(c) for i in range(r)]
        return Matrix((c, r), data=data_t)

    def _minor(self, i: int, j: int) -> 'Matrix':
        """Get the minor matrix by removing row i and column j"""
        r, c = self.shape
        data = []
        for ri in range(r):
            if ri != i:
                for ci in range(c):
                    if ci != j:
                        data.append(self[ri, ci])
        return Matrix((r-1, c-1), data=data)

    def det(self) -> float:
        """Calculate determinant using cofactor expansion"""
        r, c = self.shape
        if r != c:
            raise ValueError("Matrix must be square to compute determinant")
        if r == 1:
            return self[0, 0]
        if r == 2:
            return self[0, 0] * self[1, 1] - self[0, 1] * self[1, 0]
        
        det = 0
        for j in range(c):
            cofactor = (-1) ** j * self[0, j] * self._minor(0, j).det()
            det += cofactor
        return det


    def _eigvalsh(self, max_iter: int = 100) -> float:
        """Compute largest eigenvalue using power iteration"""
        n = self.shape[0]
        b = Vector(n, data=[random.random() for _ in range(n)])
        norm = math.sqrt(sum(x**2 for x in b))
        b = Vector(n, data=[x / norm for x in b])

        for _ in range(max_iter):
            y = self @ b
            norm = math.sqrt(sum(x**2 for x in y))
            if norm == 0: break
            b = Vector(n, data=[x / norm for x in y])

        Ab = self @ b
        return sum(b[i] * Ab[i] for i in range(n))

    def norm(self, taste: str = 'euclidean') -> float:
        """Calculate matrix norm"""
        if taste == 'euclidean':
            AtA = self.t() @ self
            eigmax = AtA._eigvalsh()
            return math.sqrt(eigmax)
        elif taste == 'abs':
            return max(sum(abs(self[i, j]) for i in range(self.shape[0])) for j in range(self.shape[1]))
        elif taste == 'max':
            return max(sum(abs(self[i, j]) for j in range(self.shape[1])) for i in range(self.shape[0]))
        else:
            raise ValueError("Invalid norm type")

    def to_graph(self, weighted=True) -> 'Graph':
        """Convert matrix to graph representation"""
        g = Graph()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                g.add_vertex(i)
                g.add_vertex(j)
                if weighted:
                    if self[i, j] != 0:
                        g.set_edge(i, j, self[i, j])
                else:
                    if self[i, j] != 0:
                        g.set_edge(i, j, 1)
        return g

class Vector(GenericTensor):
    def __new__(cls, shape: tuple|int=None, command: str = 'zeros', data: List[float] = None) -> "Vector":
        if isinstance(shape, tuple) and len(shape) == 1:
            shape = shape[0]
        shape = len(data) if shape is None else shape
        return super().__new__(cls, (shape,), command, data)

    def __add__(self, B) -> 'Vector':
        result = super().__add__(B)
        return Vector(result.shape, data=list(result))

    def __mul__(self, B: Union[float, int, "Vector"]) -> 'Vector':
        result = super().__mul__(B)
        return Vector(result.shape, data=list(result))

    @classmethod
    def __reshape__(cls, T: "Vector" = None) -> 'Vector':
        result = super().__reshape__(T)
        return Vector(result.shape, data=list(result))

    def __matmul__(self, B: "GenericTensor") -> "GenericTensor":
        result, out_shape = self._matmul_core(B)
        if len(out_shape) == 1:  # Vector
            return Vector(out_shape[0], data=result)
        elif len(out_shape) == 2:  # Matrix
            return Matrix(out_shape, data=result)
        else:  # Tensor
            return Tensor(out_shape, data=result)

    def norm(self, taste: str = 'euclidean') -> float:
        """Calculate vector norm"""
        if taste == 'euclidean':  # ||x||₂
            return math.sqrt(sum(v**2 for v in self))
        elif taste == 'abs':  # ||x||₁
            return sum(abs(v) for v in self)
        elif taste == 'max':  # ||x||∞
            return max(abs(v) for v in self)
        else:
            raise ValueError("Not permitted")
        
    def __str__(self):
        return "[" + " ".join(f"{x:.2f}" for x in self) + "]"


class Pair(tuple):
    """Class to represent and compare edge keys (u, v)"""
    def __new__(cls, i, j):
        return super().__new__(cls, (i, j))
    @property
    def i(self):  return self[0]
    @property
    def j(self): return self[1]
    def __eq__(self, other) -> bool:
        return tuple(self) == tuple(other)
    def __lt__(self, other) -> bool:
        if self.i != other.i: return self.i < other.i
        else: return self.j < other.j
    def __hash__(self): return super().__hash__()
    def __repr__(self): return f"({self.i}, {self.j})"


class GraphMixin:
    """Base mixin class for graph operations"""

    def __init__(self, n: int = 0):
        self.n = 0  # Number of vertices
        self.d = 0  # Density
        self.V = {}  # Vertex 
        self.E = {}  # Edge {(u, v): {'weight': w}}
        if isinstance(n, int) and n > 0:
            for v in range(n): 
                self.add_vertex(v)

    def __len__(self): return len(self.V) if self.n == 0 or self.n == None else self.n
    def __update_density(self): self.d = len(self.E) / len(self) if self.V else 0
    def __update_size(self, delta: int): self.n += delta ;  self.__update_density()
    def __add_edge_data(self, edge_key:Pair, data: dict):
        if edge_key in self.E:
            self.E[edge_key].update(data)
        else:
            self.E[edge_key] = data
    def __create_pair(self,i,j): return Pair(i, j)

    def __add_arc(self, edge_key:Pair, 
                  w: float, 
                  overwrite: bool):
        data = {'weight': w}
        if overwrite or edge_key not in self.E: self.E[edge_key] = data
        else: self.__add_edge_data(edge_key,data)
        self.__update_density()

    def __get_edge_data(self, edge_key:Pair,key:str):
        return self.E[edge_key][key] if edge_key in self else None

    def add_vertex(self, v: int, tag: str = ''):
        if v not in self.V: self.V[v] = tag ; self.__update_size(1)

    def get_edge_weight(self, u: int, v: int , ) -> float:
        edge_key = self.__create_pair(u, v)
        w=self.__get_edge_data(edge_key,'weight')
        return w if w  else 0.0 # zero is none

    def __iter__(self):
        for key in self.E.keys():  yield key

    def set_edge(self, u: int, v: int, w: float = 1.0, directed: bool = True, overwrite: bool = False):
        edge_key = self.__create_pair(u, v)
        if directed: self.__add_arc(edge_key, w, overwrite)
        else:
            self.__add_arc(edge_key, w, overwrite)
            rev_key = self.__create_pair(v, u)
            self.__add_arc(rev_key, w, overwrite)
        return edge_key
        

    def del_edge(self, u: int, v: int, directed: bool = True):
        edge_key = self.__create_pair(u, v)
        if edge_key in self.E: 
            del self.E[edge_key]
            if not directed:  
                rev_key = self.__create_pair(v, u)
                if rev_key in self.E:
                    del self.E[rev_key]
        self.__update_density()

    def del_vertex(self, d: int):
        """Delete vertex and its incident edges"""
        if d in self.V:
            del self.V[d] ; self.__update_size(-1)
        to_del = [ t for t in self if d in t]
        for t in to_del:
            del self.E[t]
        self.__update_density()

    def allocate_vertex(self, lb: int = 0, ub: int = 1):
        for i in range(lb, ub):  self.add_vertex(i)

    def t(self):
        """Return transpose of graph"""
        Gt = self.__class__()
        Gt.V = self.V.copy()
        Gt.n = self.n
        Gt.E = {self.__create_pair(v, u): self.E[Pair(u, v)].copy() for u, v in self}
        return Gt

    def to_square_matrix(self, weighted: bool = True):
        n = len(self)
        mat = Matrix((n, n), command='zeros')
        for k in self.V:
            for u,v in self:
                if u == k: 
                    mat[k, v] = self.get_edge_weight(u, v) if weighted else 1.0
        return mat

    def __str__(self) -> str:
        out = f"Graph - Size: {self.n}, Density: {self.d:.2f}\n"
        out += "Vertices:\n"
        for k, tag in self.V.items():
            out += f"{tag if tag else k}"
            for u,v in self:
                edlis=self.E[(u,v)]
                if u == k:
                    term = self.V[v] if self.V[v] else v
                    if edlis:
                        out += f" --({edlis})--> {term}"
                    else:
                        out += f" --> {term}"
            out += "\n"
        return out

    def _get_rank(self , kind : str=None) -> dict:
        kind = 'i' if not kind else kind
        rank = {v: 0 for v in self.V}
        for (u, v) in self:
            if kind=='i': rank[v] += 1 # input degree
            elif kind=='o': rank[u] += 1 #  output degree
            elif kind=='b': rank[u] += 1;rank[v] += 1 # both degree
            else:
                raise ValueError("Unsupported kind")
        return rank

    def _get_neighbors(self, s: int) -> list: return [v for u,v in self if u == s]

    def _explore_graph(self, s: int, 
                            mode:str,
                            b_l: dict,
                            pre: dict, 
                            post_order: list,
                            glo_var: int = 0) :
        """Explore graph using DFS or BFS mode"""
        # Initialize data structures based on mode
        if mode == 'edge':
            # Initialize pre and post_order for all vertices
            pass
        elif mode == 'dfs':

            b_var = False
            if s not in pre:
                pre[s] = 0
        elif mode == 'bfs':
            if s not in pre:
                pre[s] = True
            b_var = True
        else:
            raise ValueError("Unsupported mode. Choose 'edge', 'bfs', or 'dfs'")

        # Initialize queue with start vertex
        deq = [(s, True)]

        #######################################
        ##          DFS Edge classification
        #######################################
        def pre_set_kind(u, v):
            # TODO: fix this output
            # {(0, 1): 'tree', (0, 2): 'tree', (0, 3): 'tree', (1, 2): 'tree', (2, 3): 'forward'}

            if pre[v] == 0: 
                kind = 'tree'
            else: 
                kind = (
                    'self'    if u == v else
                    'back'    if pre[u] > pre[v] and post_order[v] == 0 else
                    'forward' if pre[u] < pre[v] and post_order[v] > 0 else
                    'cross'
                )
            edge_key = self.__create_pair(u, v)
            b_l[edge_key] = kind
            self.__add_edge_data(edge_key, {"kind": kind})
            if kind == 'tree':  
                return True # non bipartito
            return False
        
        def post_set_kind(u):
            nonlocal glo_var
            glo_var += 1
            post_order[u] = glo_var

        #######################################
        ##           BFS bipartite 
        #######################################
        def pre_set_bipartite(u, v):
            nonlocal b_var
            if s not in b_l:
                b_l[u] = 0 
            if v not in pre:
                pre[v] = True
                b_l[v] = 1 - b_l[u]
                if v not in post_order:
                    post_order.append(v)
                return True
            elif b_l[v] == b_l[u]:
                b_var = False # non bipartito
            return False
        
        def post_set_bipartite(u):
            if u not in post_order:
                post_order.append(u)

        #######################################
        ##           DFS visit 
        #######################################
        def pre_set_visit(u, v):
            if v not in pre:
                pre[v] = True
                return True
            else:
                nonlocal b_var
                b_var = True
                return False
        
        def post_set_visit(u):
            if u not in post_order:
                post_order.append(u)

        #######################################
        # fun mode selection
        if mode == 'edge':
            pre_fun = pre_set_kind
            post_fun = post_set_kind
        elif mode == 'dfs':
            pre_fun = pre_set_visit
            post_fun = post_set_visit
        elif mode == 'bfs':
            pre_fun = pre_set_bipartite
            post_fun = post_set_bipartite

        #######################################
        # exploration loop
        while deq:
            u, visited = deq.pop(0) if mode == "bfs" else deq.pop()
            if visited:
                if pre[u] == 0:
                    glo_var += 1
                    pre[u] = glo_var
                deq.append((u, False))
                # collect neighbors
                neighbors = [(v, True) for x, v in self if x == u and pre_fun(u, v)]
                rev_neighbors = neighbors[::-1]
                deq += neighbors if mode == "bfs" else rev_neighbors
            else:
                post_fun(u)

        #######################################
        # returns
        if mode == 'edge': 
            return glo_var
        else:
            return pre, post_order, b_var





class Graph(GraphMixin):
    """Basic graph implementation"""

    def _relax_edge(self, u: int, v: int, dist: dict, prev: dict) -> bool:
        """Relax edge for shortest path algorithms"""
        w=self.get_edge_weight(u,v) 
        if dist[u] + w < dist[v]: # 0 + (0+inf)
            dist[v] = dist[u] + w
            prev[v] = u
            return True
        return False

    def _graph_cover_forest(self) -> tuple:
        pre = {}
        post_order = []
        for v in self.V:
            if v not in pre:
                _pre, post_insert, _ = self._explore_graph(v, 'dfs', {}, pre, [])
                if post_insert:
                    post_order.append(post_insert)
        return pre, post_order

    def _connected_components(self):
        return self._graph_cover_forest()[1]

    def _strongly_connected_components(self):
        Gt = self.t()
        scc_list = []
        # Get the full post-order for all components
        pre = {}
        post_order = []
        for v in self.V:
            if v not in pre:
                _pre, post_extension, _ = self._explore_graph(v, 'dfs', {}, pre, [])
                post_order+=post_extension[::-1]

        # Reset visited for second DFS
        visited = {}
        # Process vertices in reverse post-order
        for v in post_order:
            if v not in visited:
                _pre, comp, _ = Gt._explore_graph(v, 'dfs', {}, visited, [])
                if comp:
                    scc_list.append(comp)
        return scc_list


#################


class WeightedGraph(Graph):
    def set_edge(self, u: int, v: int, w: float = 1.0, directed: bool = True, overwrite: bool = False):
        super().set_edge(u, v, w, directed, overwrite)

class UndirectedGraph(Graph):
    def set_edge(self, u: int, v: int, w: float = 1.0, overwrite: bool = False):
        super().set_edge(u, v, w, False, overwrite)



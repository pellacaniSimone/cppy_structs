import unittest
import math
import random
import os
import tempfile
from typing import List 
from tengraph import (Tensor, Vector, Matrix, Graph, WeightedGraph,UndirectedGraph,
 GenericTensor,
    GraphMixin
    )

from collections import defaultdict




#############################################################################
#----------------------------------------------------------------------------
#############################################################################
#                  Tensor tests
#############################################################################
#----------------------------------------------------------------------------
#############################################################################


# Common test data and utilities
class TestData:
    @staticmethod
    def create_test_matrix(rows: int, cols: int, data: List[float] = None) -> Matrix:
        return Matrix((rows, cols), data=data or [1.0] * (rows * cols))
    
    @staticmethod
    def create_test_vector(size: int, data: List[float] = None) -> Vector:
        return Vector(size, data=data or [1.0] * size)
    
    @staticmethod
    def create_test_graph() -> Graph:
        g = Graph(3)
        g.set_edge(0, 1, 1.0, 'a')
        g.set_edge(1, 2, 2.0, 'b')
        return g

class TestTensorOperations(unittest.TestCase):
    """Test suite for basic tensor operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.A = Matrix((2, 3), command='ones')
        self.B = Matrix((3, 4), command='ones')
        self.v = Vector(3, command='rand')
        
    def test_matrix_multiplication(self):
        """Test matrix multiplication operations"""
        C = self.A @ self.B
        self.assertEqual(C.shape, (2, 4))
        # Check that each element is 3.0 (sum of ones)
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                self.assertEqual(C[i, j], 3.0)
        
    def test_vector_operations(self):
        """Test vector operations and norms"""
        self.assertEqual(len(self.v), 3)
        self.assertGreaterEqual(self.v.norm(), 0)
        self.assertGreaterEqual(self.v.norm('abs'), 0)
        self.assertGreaterEqual(self.v.norm('max'), 0)
        
    def test_tensor_reshape(self):
        """Test tensor reshaping operations"""
        T = Tensor((2, 3), command='ones')
        reshaped = T.__reshape__(T)
        self.assertEqual(reshaped.shape, (2, 3))
        self.assertTrue(all(x == 1.0 for x in reshaped))



class TestTensorCreation(unittest.TestCase):
    def setUp(self):
        self.test_data = TestData()
    
    def test_tensor_creation_zeros(self):
        """Test creating tensors with zeros"""
        shapes = [(2, 3), (5,), (2, 3, 4)]
        for shape in shapes:
            t = Tensor(shape, command="zeros")
            self.assertEqual(t.shape, shape)
            self.assertEqual(list(t), [0.0] * math.prod(shape))
    
    def test_tensor_creation_ones(self):
        """Test creating tensors with ones"""
        t = Tensor([2, 3], command="ones")
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(list(t), [1.0] * 6)
    
    def test_tensor_creation_from_data(self):
        """Test creating tensors from data"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        t = Tensor([2, 3], data=data)
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(list(t), data)
    
    def test_tensor_creation_rand(self):
        """Test creating tensors with random values"""
        random.seed(42)
        t = Tensor([2, 3], command="rand")
        self.assertEqual(t.shape, (2, 3))
        self.assertTrue(all(0 <= x <= 10 for x in t))
    
    def test_tensor_creation_randn(self):
        """Test creating tensors with normally distributed random values"""
        random.seed(42)
        t = Tensor([2, 3], command="randn")
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(len(list(t)), 6)
    
    def test_invalid_creation(self):
        """Test error handling for invalid tensor creation"""
        with self.assertRaises(ValueError):
            Tensor([0, 3])
        with self.assertRaises(ValueError):
            Tensor([-1, 3])
        with self.assertRaises(ValueError):
            Tensor([2, 3], command="invalid")
        with self.assertRaises(ValueError):
            Tensor([2, 3], data=[1.0, 2.0, 3.0])

class TestTensorBasicOperations(unittest.TestCase):
    def setUp(self):
        self.t1 = Tensor([2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.t2 = Tensor([2, 3], data=[6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        self.scalar = 2.0
    
    def test_indexing(self):
        """Test tensor indexing"""
        self.assertEqual(self.t1[0][0], 1.0)
        self.assertEqual(self.t1[1][0], 4.0)
        self.assertEqual(self.t1[0, 0], 1.0)
        self.assertEqual(self.t1[0, 1], 2.0)
        self.assertEqual(self.t1[1, 2], 6.0)
        
        with self.assertRaises(IndexError):
            self.t1[2][0]
        with self.assertRaises(IndexError):
            self.t1[0][3]
        with self.assertRaises(IndexError):
            self.t1[0, 0, 0]
    
    def test_setitem(self):
        """Test setting tensor values"""
        t = Tensor([2, 3], command="zeros")
        t[0, 0] = 5.0
        self.assertEqual(t[0][0], 5.0)
        t[1, 2] = 7.0
        self.assertEqual(t[1, 2], 7.0)
        
        with self.assertRaises(IndexError):
            t[2, 0] = 10.0
    
    def test_addition(self):
        """Test tensor addition"""
        result = self.t1 + self.t2
        expected = [7.0] * 6
        self.assertEqual(list(result), expected)
        self.assertEqual(result.shape, (2, 3))
        
        t3 = Tensor([3, 2])
        with self.assertRaises(ValueError):
            _ = self.t1 + t3
    
    def test_scalar_multiplication(self):
        """Test tensor multiplication with scalar"""
        result = self.t1 * self.scalar
        expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        self.assertEqual(list(result), expected)
        self.assertEqual(result.shape, (2, 3))
    
    def test_element_wise_multiplication(self):
        """Test element-wise tensor multiplication"""
        result = self.t1 * self.t2
        expected = [6.0, 10.0, 12.0, 12.0, 10.0, 6.0]
        self.assertEqual(list(result), expected)
        self.assertEqual(result.shape, (2, 3))
        
        t3 = Tensor([3, 2])
        with self.assertRaises(ValueError):
            _ = self.t1 * t3
    
    def test_len(self):
        """Test tensor length"""
        self.assertEqual(len(self.t1), 6)
    
    def test_str(self):
        """Test tensor string representation"""
        t = Tensor([2, 2], data=[1.0, 2.0, 3.0, 4.0])
        s = str(t)
        for val in [1.0, 2.0, 3.0, 4.0]:
            self.assertIn(f"{val:.2f}", s)

class TestTensorMatrixMultiplication(unittest.TestCase):
    def setUp(self):
        self.A = Tensor([2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.B = Tensor([3, 2], data=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        self.v = Tensor([3], data=[7.0, 8.0, 9.0])

    def test_matrix_matrix_multiplication(self):
        """Test multiplication between two matrices"""
        C = self.A @ self.B
        self.assertEqual(C.shape, (2, 2))
        self.assertEqual(list(C), [58.0, 64.0, 139.0, 154.0])
    
    def test_matrix_vector_multiplication(self):
        """Test multiplication between a matrix and a vector"""
        C = self.A @ self.v
        self.assertEqual(C.shape, (2,))
        self.assertEqual(list(C), [50.0, 122.0])
    def test_vector_matrix_multiplication(self):
        """Test multiplication between a vector and a matrix"""
        v = Tensor([2], data=[1.0, 2.0])
        C =  self.B @ v
        self.assertEqual(C.shape, (3,))
        self.assertEqual(list(C), [23.0, 29.0, 35.0])
    
    def test_vector_vector_multiplication(self):
        """Test dot product between two vectors"""
        v1 = Tensor([3], data=[1.0, 2.0, 3.0])
        v2 = Tensor([3], data=[4.0, 5.0, 6.0])
        C = v1 @ v2
        self.assertEqual(C.shape, (1,))
        self.assertEqual(list(C), [32.0])
    
    def test_batch_matmul(self):
        """Test batch matrix multiplication with 3D tensors"""
        A = Tensor([2, 2, 3], data=[
            1.0, 2.0, 3.0, 4.0, 1.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 1.0, 12.0
        ])
        B = Tensor([3, 2, 2], data=[
            1.0, 2.0, 3.0, 4.0, 5.0, 1.0,
            1.0, 8.0, 1.0, 1.0, 1.0, 12.0
        ])
        C = A @ B
        self.assertEqual(C.shape, (2, 2, 2, 2))
        expected = [
            14.0, 7.0, 8.0, 56.0, 15.0, 15.0, 19.0, 96.0,
            56.0, 31.0, 38.0, 200.0, 27.0, 33.0, 43.0, 192.0
        ]
        self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(C, expected)))
    
    def test_higher_dimensional_matmul(self):
        """Test matrix multiplication with 4D tensors"""
        A = Tensor([2, 1, 2, 3], data=[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        ])
        B = Tensor([3, 4], data=[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        ])
        C = A @ B
        self.assertEqual(C.shape, (2, 1, 2, 4))
        expected = [
            38.0, 44.0, 50.0, 56.0,
            83.0, 98.0, 113.0, 128.0
        ]
        self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(C, expected)))
    
    def test_invalid_matmul_shapes(self):
        """Test error handling for invalid shapes in matrix multiplication"""
        A = Tensor([2, 3])
        B = Tensor([4, 2])
        with self.assertRaises(ValueError):
            _ = A @ B




class TestGenericTensor(unittest.TestCase):
    def setUp(self):
        self.tensor = GenericTensor((2, 3), command='ones')
    
    
    def test_fill_triangle(self):
        """Test triangle filling methods"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        upper = GenericTensor._fill_triangle(data, (2, 3), lambda i: range(i, 3))
        lower = GenericTensor._fill_triangle(data, (2, 3), lambda i: range(i+1))
        self.assertEqual(len(upper), 6)
        self.assertEqual(len(lower), 6)
    
    def test_diag_matrix(self):
        """Test diagonal matrix creation"""
        diag = GenericTensor._diag_matrix((3, 3), [1.0, 2.0, 3.0])
        self.assertEqual(diag[0], 1.0)
        self.assertEqual(diag[4], 2.0)
        self.assertEqual(diag[8], 3.0)
    
    def test_sparse_matrix(self):
        """Test sparse matrix creation"""
        sparse = GenericTensor._sparse_matrix((3, 3), 0.5)
        self.assertEqual(len(sparse), 9)
        non_zero = sum(1 for x in sparse if x != 0)
        self.assertGreater(non_zero, 0)

class TestTensorExtended(unittest.TestCase):
    def setUp(self):
        self.tensor = Tensor((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    def test_reshape(self):
        """Test tensor reshaping"""
        reshaped = Tensor.__reshape__(self.tensor)
        self.assertEqual(reshaped.shape, (2, 3))
    
    def test_save_load(self):
        """Test tensor save and load operations"""
        with tempfile.NamedTemporaryFile() as tmp:
            self.tensor.save(tmp.name)
            loaded = Tensor.load(tmp.name)
            self.assertEqual(loaded.shape, self.tensor.shape)
            self.assertEqual(list(loaded), list(self.tensor))
    
    def test_elementwise_operations(self):
        """Test elementwise operations"""
        other = Tensor((2, 3), data=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        add_result = self.tensor + other
        sub_result = self.tensor - other
        mul_result = self.tensor * other
        self.assertEqual(list(add_result), [3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.assertEqual(list(sub_result), [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(list(mul_result), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0])











class TestMatrixVectorClasses(unittest.TestCase):
    def setUp(self):
        self.test_data = TestData()
    
    def test_matrix_creation(self):
        """Test Matrix class creation"""
        M = Matrix((2, 3), command="ones")
        self.assertEqual(M.shape, (2, 3))
        self.assertEqual(list(M), [1.0] * 6)
        self.assertEqual(M.shape[0], 2)
        self.assertEqual(M.shape[1], 3)
    
    def test_matrix_transpose(self):
        """Test matrix transpose operation"""
        M = Matrix((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        Mt = M.t()
        self.assertEqual(Mt.shape, (3, 2))
        self.assertEqual(Mt.shape[0], 3) # rows
        self.assertEqual(Mt.shape[1], 2) # cols
        self.assertEqual(Mt[0, 0], 1.0)
        self.assertEqual(Mt[1, 0], 2.0)
        self.assertEqual(Mt[2, 0], 3.0)
        self.assertEqual(Mt[0, 1], 4.0)
        self.assertEqual(Mt[1, 1], 5.0)
        self.assertEqual(Mt[2, 1], 6.0)
    
    def test_vector_creation(self):
        """Test Vector class creation"""
        v = Vector(3, command="ones")
        self.assertEqual(v.shape, (3,))
        self.assertEqual(list(v), [1.0, 1.0, 1.0])
    
    def test_vector_norm(self):
        """Test vector norm calculations"""
        v = Vector(3, data=[3.0, 4.0, 12.0])
        self.assertAlmostEqual(v.norm(), 13.0)
        self.assertEqual(v.norm('abs'), 19.0)
        with self.assertRaises(ValueError):
            v.norm('invalid')
    
    def test_matrix_vector_operations(self):
        """Test combined operations between matrices and vectors"""
        M = Matrix((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v = Vector(3, data=[7.0, 8.0, 9.0])
        result = M @ v
        self.assertTrue(isinstance(result, Matrix))
        self.assertEqual(result.shape, (2,))
        self.assertEqual(list(result), [50.0, 122.0])

class TestTensorIO(unittest.TestCase):
    def setUp(self):
        self.test_tensor = Tensor([2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_save_load(self):
        """Test saving and loading a tensor"""
        filepath = os.path.join(self.temp_dir.name, "test_tensor.txt")
        self.test_tensor.save(filepath)
        loaded_tensor = Tensor.load(filepath)
        self.assertEqual(loaded_tensor.shape, self.test_tensor.shape)
        self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(loaded_tensor, self.test_tensor)))
    
    def test_load_nonexistent_file(self):
        """Test error handling when loading a non-existent file"""
        filepath = os.path.join(self.temp_dir.name, "nonexistent.txt")
        with self.assertRaises(FileNotFoundError):
            Tensor.load(filepath)

class TestIntegrationScenarios(unittest.TestCase):
    def setUp(self):
        self.test_data = TestData()
    
    def test_chained_operations(self):
        """Test chaining multiple tensor operations"""
        A = Matrix((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        B = Matrix((3, 2), data=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        v = Vector(data=[1.0, -1.0])
        result = (A @ B) @ v
        expected = [-6.0, -15.0]
        self.assertEqual(list(result), expected)
    
    def test_mixed_dimensional_operations(self):
        """Test operations mixing tensors of different dimensions"""
        A = Tensor([2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        B = Tensor([3, 1, 4], data=[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        ])
        C = A @ B
        self.assertEqual(C.shape, (2, 1, 4))
        expected = [
            38.0, 44.0, 50.0, 56.0,
            83.0, 98.0, 113.0, 128.0
        ]
        self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(C, expected)))
    
    def test_performance_large_tensors(self):
        """Test performance with larger tensors (basic benchmark)"""
        import time
        size = 50
        A = Tensor([size, size], command="rand")
        B = Tensor([size, size], command="rand")
        start_time = time.time()
        C = A @ B
        end_time = time.time()
        self.assertEqual(C.shape, (size, size))


class TestMatrixOperationsExtended(unittest.TestCase):
    def setUp(self):
        self.matrix = Matrix((2, 2), data=[1.0, 2.0, 3.0, 4.0])
    
    def test_matrix_to_graph(self):
        """Test matrix to graph conversion"""
        g = self.matrix.to_graph()
        self.assertIsInstance(g, Graph)
        self.assertEqual(g.n, 2)
    
    def test_matrix_norm_edge_cases(self):
        """Test matrix norm edge cases"""
        zero_matrix = Matrix((2, 2), command='zeros')
        self.assertEqual(zero_matrix.norm('euclidean'), 0)
        self.assertEqual(zero_matrix.norm('abs'), 0)
        self.assertEqual(zero_matrix.norm('max'), 0)
        
        with self.assertRaises(ValueError):
            self.matrix.norm('invalid')















#############################################################################
#----------------------------------------------------------------------------
#############################################################################
#                  Graph tests
#############################################################################
#----------------------------------------------------------------------------
#############################################################################




class TestGraphMixinBase(unittest.TestCase):
    """Base test suite for edge classification in graphs"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gm = GraphMixin()  # Initialize the graph object
        random.seed(42)

        self.gm.add_vertex(1)
        self.gm.add_vertex(2)
        self.gm.add_vertex(3)
        self.gm.set_edge(2, 1, 1.0,)
        self.gm.set_edge(1, 3, 2.0,)
        self.gm.set_edge(1, 2, 3.0,)

    def test_str(self):
        """Test string representation of graph"""
        expected_str = """Graph - Size: 3, Density: 1.00
Vertices:
1 --({'weight': 2.0})--> 3 --({'weight': 3.0})--> 2
2 --({'weight': 1.0})--> 1
3
"""
        print(self.gm)
        self.assertEqual(str(self.gm), expected_str)




class TestGraphOperations(unittest.TestCase):
    """Test suite for graph operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.g = Graph()

        
    def test_basic_graph_operations(self):
        """Test basic graph operations"""
        self.g.add_vertex(1, "A")
        self.g.add_vertex(2, "B")
        self.g.set_edge(1, 2, 5.0,  directed=False)
        self.assertEqual(self.g.n, 2)
        # For undirected edges, both (u,v) and (v,u) are stored
        self.assertEqual(len(self.g.E), 2)
        # Test edge data access
        edge_key = self.g._GraphMixin__create_pair(1, 2)
        self.assertEqual(self.g.E[edge_key]['weight'], 5.0)

        


class TestWeightedGraph(unittest.TestCase):
    def setUp(self):
        self.wg = WeightedGraph()
        for i in range(4):
            self.wg.add_vertex(i)
    
    def test_weights_not_none(self):
        """Test that only positive weights are allowed"""
        self.wg.set_edge(0, 1, 5.0,)
        self.assertEqual(self.wg.E[(0, 1)]['weight'], 5.0)
        

class TestUndirectedGraph(unittest.TestCase):
    def setUp(self):
        self.ug = UndirectedGraph(4)

    
    def test_undirected_edges(self):
        """Test that edges are always undirected"""
        self.ug.set_edge(0, 1, 1.0, )
        kp=self.ug._GraphMixin__create_pair(0, 1)
        self.assertIn(kp, self.ug.E)
        kp=self.ug._GraphMixin__create_pair(1, 0)
        self.assertIn(kp, self.ug.E)
        # Test that directed parameter is ignored
        self.ug.set_edge(1, 2, 2.0, )
        kp=self.ug._GraphMixin__create_pair(1, 2)
        self.assertIn(kp, self.ug.E)
        kp=self.ug._GraphMixin__create_pair(2, 1)
        self.assertIn((2, 1), self.ug.E)
        # Check edge weights are the same in both directions
        self.assertEqual(self.ug.E[(0, 1)]['weight'], 1.0)
        self.assertEqual(self.ug.E[(1, 0)]['weight'], 1.0)
        self.assertEqual(self.ug.E[(1, 2)]['weight'], 2.0)
        self.assertEqual(self.ug.E[(2, 1)]['weight'], 2.0)










if __name__ == "__main__":
    unittest.main(verbosity=2)
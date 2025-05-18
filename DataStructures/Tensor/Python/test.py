import unittest
import math
import random
import os
import tempfile
from typing import List
from algebra import Tensor,Vector,Matrix

class TestTensorCreation(unittest.TestCase):
    """Test tensor initialization with different shapes and commands"""
    
    def test_tensor_creation_zeros(self):
        """Test creating tensors with zeros"""
        t = Tensor([2, 3], command="zeros")
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(list(t), [0.0] * 6)
        
        # Test 1D tensor
        t = Tensor([5], command="zeros")
        self.assertEqual(t.shape, (5,))
        self.assertEqual(list(t), [0.0] * 5)
        
        # Test 3D tensor
        t = Tensor([2, 3, 4], command="zeros")
        self.assertEqual(t.shape, (2, 3, 4))
        self.assertEqual(list(t), [0.0] * 24)

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
        random.seed(42)  # For reproducibility
        t = Tensor([2, 3], command="rand")
        self.assertEqual(t.shape, (2, 3))
        # Check values are within expected range (0 to 10)
        self.assertTrue(all(0 <= x <= 10 for x in t))
        
    def test_tensor_creation_randn(self):
        """Test creating tensors with normally distributed random values"""
        random.seed(42)  # For reproducibility
        t = Tensor([2, 3], command="randn")
        self.assertEqual(t.shape, (2, 3))
        # Simply check that values are created (hard to test exact values with random)
        self.assertEqual(len(list(t)), 6)
        
    def test_invalid_creation(self):
        """Test error handling for invalid tensor creation"""
        # Invalid shape dimensions
        with self.assertRaises(ValueError):
            Tensor([0, 3])
            
        with self.assertRaises(ValueError):
            Tensor([-1, 3])
            
        # Invalid command
        with self.assertRaises(ValueError):
            Tensor([2, 3], command="invalid")
            
        # Data length mismatch
        with self.assertRaises(ValueError):
            Tensor([2, 3], data=[1.0, 2.0, 3.0])  # Should be 6 elements


class TestTensorBasicOperations(unittest.TestCase):
    """Test basic tensor operations like indexing, addition, multiplication"""
    
    def setUp(self):
        # Create some test tensors
        self.t1 = Tensor([2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.t2 = Tensor([2, 3], data=[6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        self.scalar = 2.0
        
    def test_indexing(self):
        """Test tensor indexing"""
        # Linear indexing
        self.assertEqual(self.t1[0], 1.0)
        self.assertEqual(self.t1[3], 4.0)
        
        # Tuple indexing
        self.assertEqual(self.t1[0, 0], 1.0)
        self.assertEqual(self.t1[0, 1], 2.0)
        self.assertEqual(self.t1[1, 2], 6.0)
        
        # Test invalid indexing
        with self.assertRaises(IndexError):
            self.t1[2, 0]  # Out of bounds
            
        with self.assertRaises(IndexError):
            self.t1[0, 3]  # Out of bounds
            
        with self.assertRaises(IndexError):
            self.t1[0, 0, 0]  # Too many indices
            
    def test_setitem(self):
        """Test setting tensor values"""
        t = Tensor([2, 3], command="zeros")
        t[0, 0] = 5.0
        self.assertEqual(t[0, 0], 5.0)
        
        t[1, 2] = 7.0
        self.assertEqual(t[1, 2], 7.0)
        
        # Test invalid setitem
        with self.assertRaises(IndexError):
            t[2, 0] = 10.0  # Out of bounds
            
    def test_addition(self):
        """Test tensor addition"""
        result = self.t1 + self.t2
        expected = [7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
        self.assertEqual(list(result), expected)
        self.assertEqual(result.shape, (2, 3))
        
        # Test invalid addition (shape mismatch)
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
        
        # Test invalid multiplication (shape mismatch)
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
        self.assertIn("1.00", s)
        self.assertIn("2.00", s)
        self.assertIn("3.00", s)
        self.assertIn("4.00", s)


class TestTensorMatrixMultiplication(unittest.TestCase):
    """Test matrix multiplication operations"""
    
    def test_matrix_matrix_multiplication(self):
        """Test multiplication between two matrices"""
        # Basic 2x3 @ 3x2 = 2x2
        A = Tensor([2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        B = Tensor([3, 2], data=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        C = A @ B
        self.assertEqual(C.shape, (2, 2))
        self.assertEqual(list(C), [58.0, 64.0, 139.0, 154.0])
        
    def test_matrix_vector_multiplication(self):
        """Test multiplication between a matrix and a vector"""
        A = Tensor([2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v = Tensor([3], data=[7.0, 8.0, 9.0])
        C = A @ v
        self.assertEqual(C.shape, (2,))  # Result should be a vector
        self.assertEqual(list(C), [50.0, 122.0])
        
    def test_vector_matrix_multiplication(self):
        """Test multiplication between a vector and a matrix"""
        v = Tensor([2], data=[1.0, 2.0])
        A = Tensor([2, 3], data=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        C = v @ A
        self.assertEqual(C.shape, (3,))  # Result should be a vector
        self.assertEqual(list(C), [27.0, 30.0, 33.0])
        
    def test_vector_vector_multiplication(self):
        """Test dot product between two vectors"""
        v1 = Tensor([3], data=[1.0, 2.0, 3.0])
        v2 = Tensor([3], data=[4.0, 5.0, 6.0])
        C = v1 @ v2
        self.assertEqual(C.shape, (1,))  # Result should be a scalar tensor
        self.assertEqual(list(C), [32.0])  # 1*4 + 2*5 + 3*6 = 32
        
    def test_batch_matmul(self):
        """Test batch matrix multiplication with 3D tensors"""
        # 2x2x3 @ 2x3x2 = 2x2x2
        A = Tensor([2, 2, 3], data=[
            1.0, 2.0, 3.0, 
            4.0, 1.0, 6.0, 
            7.0, 8.0, 9.0, 
            10.0, 1.0, 12.0  
        ])
        B = Tensor([3, 2, 2], data=[
            1.0, 2.0,  
            3.0, 4.0,  
            5.0, 1.0,  
            1.0, 8.0,  
            1.0, 1.0, 
            1.0, 12.0  
        ])
        C = A @ B
        self.assertEqual(C.shape, (2, 2, 2, 2))
        expected = [
            14.00, 7.00,  
            8.00, 56.00,  
            15.00, 15.00 , 
            19.00, 96.00, 
            56.00, 31.00,  
            38.00, 200.00,  
            27.00, 33.00 , 
            43.00, 192.00, 
        ]
        self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(C, expected)))
        
    def test_higher_dimensional_matmul(self):
        """Test matrix multiplication with 4D tensors"""
        # Create a 2x1x2x3 tensor
        A = Tensor([2, 1, 2, 3], data=[
            1.0, 2.0, 3.0,  # batch 0, subbatch 0, row 0
            4.0, 5.0, 6.0,  # batch 0, subbatch 0, row 1
            7.0, 8.0, 9.0,  # batch 1, subbatch 0, row 0
            10.0, 11.0, 12.0  # batch 1, subbatch 0, row 1
        ])
        # Create a 3x4 tensor
        B = Tensor([3, 4], data=[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        ])
        C = A @ B
        self.assertEqual(C.shape, (2, 1, 2, 4))
        expected = [
            38.0, 44.0, 50.0, 56.0,  # batch 0, subbatch 0, row 0
            83.0, 98.0, 113.0, 128.0,  # batch 0, subbatch 0, row 1
            128.0, 152.0, 176.0, 200.0,  # batch 1, subbatch 0, row 0
            173.0, 206.0, 239.0, 272.0  # batch 1, subbatch 0, row 1
        ]
        self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(C, expected)))
        
    def test_invalid_matmul_shapes(self):
        """Test error handling for invalid shapes in matrix multiplication"""
        A = Tensor([2, 3])
        B = Tensor([4, 2])
        with self.assertRaises(ValueError):
            _ = A @ B


class TestMatrixVectorClasses(unittest.TestCase):
    """Test the Matrix and Vector specific classes"""
    
    def test_matrix_creation(self):
        """Test Matrix class creation"""
        M = Matrix((2, 3), command="ones")
        self.assertEqual(M.shape, (2, 3))
        self.assertEqual(list(M), [1.0] * 6)
        self.assertEqual(M.rows, 2)
        self.assertEqual(M.cols, 3)
        
    def test_matrix_transpose(self):
        """Test matrix transpose operation"""
        M = Matrix((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        Mt = M.t()
        self.assertEqual(Mt.shape, (3, 2))
        self.assertEqual(Mt.rows, 3)
        self.assertEqual(Mt.cols, 2)
        # Check if transposed correctly
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
        # Euclidean norm: sqrt(3² + 4² + 12²) = sqrt(9 + 16 + 144) = sqrt(169) = 13
        self.assertAlmostEqual(v.norm(), 13.0)
        
        # Absolute norm: |3| + |4| + |12| = 19
        self.assertEqual(v.norm('abs'), 19.0)
        
        # Test invalid norm
        with self.assertRaises(ValueError):
            v.norm('invalid')
            
    def test_matrix_vector_operations(self):
        """Test combined operations between matrices and vectors"""
        M = Matrix((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v = Vector(3, data=[7.0, 8.0, 9.0])
        
        # Matrix @ Vector
        result = M @ v
        self.assertTrue(isinstance(result, Tensor))
        self.assertEqual(result.shape, (2,))
        self.assertEqual(list(result), [50.0, 122.0])


class TestTensorIO(unittest.TestCase):
    """Test tensor save and load operations"""
    
    def setUp(self):
        self.test_tensor = Tensor([2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_save_load(self):
        """Test saving and loading a tensor"""
        filepath = os.path.join(self.temp_dir.name, "test_tensor.txt")
        
        # Save the tensor
        self.test_tensor.save(filepath)
        
        # Load the tensor back
        loaded_tensor = Tensor.load(filepath)
        
        # Check if the loaded tensor is correct
        self.assertEqual(loaded_tensor.shape, self.test_tensor.shape)
        self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(loaded_tensor, self.test_tensor)))
        
    def test_load_nonexistent_file(self):
        """Test error handling when loading a non-existent file"""
        filepath = os.path.join(self.temp_dir.name, "nonexistent.txt")
        with self.assertRaises(FileNotFoundError):
            Tensor.load(filepath)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complex tensor operations"""
    
    def test_chained_operations(self):
        """Test chaining multiple tensor operations"""
        A = Matrix((2, 3), data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        B = Matrix((3, 2), data=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        v = Vector( data=[1.0, -1.0])
        
        # (A @ B) * v - complex operation chain
        result = (A @ B) @ v
        
        # Expected: first A @ B = [[58, 64], [139, 154]]
        # Then element-wise multiply with v = [1, 2]
        # [[58*1, 64*2], [139*1, 154*2]] = [[58, 128], [139, 308]]
        expected = [-6.0, -15.0]
        self.assertEqual(list(result), expected)
        
    def test_mixed_dimensional_operations(self):
        """Test operations mixing tensors of different dimensions"""
        # 2x3 matrix
        A = Tensor([2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        # 3x1x4 tensor
        B = Tensor([3, 1, 4], data=[
            1.0, 2.0, 3.0, 4.0,  # dim 0, idx 0
            5.0, 6.0, 7.0, 8.0,  # dim 0, idx 1
            9.0, 10.0, 11.0, 12.0  # dim 0, idx 2
        ])
        
        # Result should be 2x1x4
        C = A @ B
        self.assertEqual(C.shape, (2, 1, 4))
        
        # Expected: A @ B calculation
        # For each output cell, multiply corresponding elements and sum
        expected = [
            # First row of A [1,2,3] @ B
            1.0*1.0 + 2.0*5.0 + 3.0*9.0,  # = 38
            1.0*2.0 + 2.0*6.0 + 3.0*10.0,  # = 44
            1.0*3.0 + 2.0*7.0 + 3.0*11.0,  # = 50
            1.0*4.0 + 2.0*8.0 + 3.0*12.0,  # = 56
            
            # Second row of A [4,5,6] @ B
            4.0*1.0 + 5.0*5.0 + 6.0*9.0,  # = 83
            4.0*2.0 + 5.0*6.0 + 6.0*10.0,  # = 98
            4.0*3.0 + 5.0*7.0 + 6.0*11.0,  # = 113
            4.0*4.0 + 5.0*8.0 + 6.0*12.0,  # = 128
        ]
        
        self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(C, expected)))
        
    def test_performance_large_tensors(self):
        """Test performance with larger tensors (basic benchmark)"""
        import time
        
        # Create large tensors for multiplication
        size = 50  # Smaller size for unit tests, adjust as needed
        A = Tensor([size, size], command="rand")
        B = Tensor([size, size], command="rand")
        
        start_time = time.time()
        C = A @ B
        end_time = time.time()
        
        # Just verify that it completes without error
        self.assertEqual(C.shape, (size, size))
        print(f"Large {size}x{size} matrix multiplication took {end_time - start_time:.4f} seconds")


from graph import Graph,DAG

class TestGraph(unittest.TestCase):
    def setUp(self):
        self.g = Graph()
        self.g.add_vertex(0)
        self.g.add_vertex(1)
        self.g.add_vertex(2)
        self.g.add_arc(0, 1, 1.0, 'a')
        self.g.add_edge(1, 2, 2.0, 'b')
        
        self.dag = DAG()
        self.dag.add_vertex(0)
        self.dag.add_vertex(1)
        self.dag.add_vertex(2)
        self.dag.add_arc(0, 1, 1.0, 'a')
        self.dag.add_arc(1, 2, 2.0, 'b')

    def test_graph_initialization(self):
        self.assertEqual(self.g.dim, 3)
        self.assertAlmostEqual(self.g.density, 3/3)
        
    def test_vertex_operations(self):
        self.g.add_vertex(3, 'new')
        self.assertEqual(self.g.dim, 4)
        self.assertIn(3, self.g.vertex)
        self.assertEqual(self.g.vertex[3], 'new')
        
        self.g.del_vertex(3)
        self.assertEqual(self.g.dim, 3)
        self.assertNotIn(3, self.g.vertex)

    def test_edge_operations(self):
        self.g.add_edge(0, 2, 3.0, 'c')
        self.assertIn((0, 2), self.g.edges)
        self.assertIn((2, 0), self.g.edges)
        
        self.g.del_edge(0, 2, directed=False)
        self.assertNotIn((0, 2), self.g.edges)
        self.assertNotIn((2, 0), self.g.edges)

    def test_transpose(self):
        gt = self.g.t()
        self.assertEqual(gt.dim, 3)
        self.assertIn((1, 0), gt.edges)
        self.assertIn((2, 1), gt.edges)

    def test_to_matrix(self):
        print(self.g)
        mat = self.g.to_matrix()
        print(mat)
        self.assertIsInstance(mat, Matrix)
        # Verifica alcuni valori della matrice
        self.assertEqual(mat[0][1], 1.0)
        self.assertEqual(mat[1][2], 2.0)
        self.assertEqual(mat[2][1], 2.0)  # Perché è un edge bidirezionale

    def test_dag_property(self):
        self.assertTrue(self.dag.is_dag())
        
        # Aggiungiamo un ciclo per testare
        self.dag.add_arc(2, 0, 3.0, 'cycle')
        self.assertFalse(self.dag.is_dag())

    def test_str_representation(self):
        s = str(self.g)
        self.assertIn("Graph - Size: 3", s)
        self.assertIn("0 --([(1.0, 'a')])--> 1", s)
        self.assertIn("1 --([(2.0, 'b')])--> 2", s)




# Run all tests
if __name__ == "__main__":
    unittest.main()

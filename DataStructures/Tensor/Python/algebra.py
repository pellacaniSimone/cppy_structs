"""

Author: Simone Pellacani (pellacanisimone2017@gmail.com)
Brief: This module provides functionality for testing linear algebra.
Version: 0.0.1
Date: 2025-05-10

Copyright (c) 2025

Refactored and document using LLMs

This module provides a foundational tensor library with support for multi-dimensional arrays,
matrix operations, and vector norms. It includes a base `GenericTensor` class and specialized
subclasses `Tensor`, `Matrix`, and `Vector` to facilitate linear algebra operations.

The `GenericTensor` class serves as the core implementation, providing functionalities for
tensor creation with various initialization commands ('zeros', 'ones', 'rand', 'randn'),
basic element-wise operations (addition, multiplication), indexing, item assignment,
string representation, and saving/loading tensors to/from files.

The `Tensor` class inherits from `GenericTensor` and acts as a general-purpose multi-dimensional
array. It ensures that arithmetic operations on `Tensor` instances return `Tensor` instances.

The `Matrix` class is a specialized subclass of `Tensor` designed for two-dimensional arrays,
commonly known as matrices. It provides convenient properties to access the number of rows and
columns and includes a method for transposing the matrix. Arithmetic operations on `Matrix`
instances return `Matrix` instances.

The `Vector` class is a specialized subclass of `Tensor` for one-dimensional arrays (or can be
viewed as a matrix with one column). It includes a method to calculate the norm of the vector
(Euclidean or absolute). Arithmetic operations on `Vector` instances return `Vector` instances.

Key Features:
    - Creation of tensors with specified shapes and initialization methods.
    - Element-wise addition and multiplication of tensors.
    - Indexing and item assignment using both linear and multi-dimensional indices.
    - Matrix multiplication (@ operator) for tensors with compatible shapes.
    - Saving and loading tensors from disk.
    - Specialized `Matrix` class with transpose functionality.
    - Specialized `Vector` class with norm calculation.
    - Maintains subclass type after arithmetic operations.

Usage:
    - Import the desired class (`Tensor`, `Matrix`, `Vector`) from the module.
    - Create instances with a specified shape and optional initialization command or data.
    - Perform arithmetic and linear algebra operations using overloaded operators and methods.

Example:
    >>> import algebra
    >>> matrix_a = algebra.Matrix( (2, 2), command='ones')
    >>> vector_b = algebra.Vector(2, data=[2.0, 3.0])
    >>> result = matrix_a @ vector_b


"""


from array import array
import random, os, math
from typing import List, Tuple, Union, 


Size = Union[List[int], Tuple[int, ...]]


class GenericTensor(array):

    @staticmethod
    def __calculate_linear_index(shape: Tuple[int, ...], indices: Tuple[int, ...]) -> int:
        index = 0
        product = 1
        for idx, dim in zip(reversed(indices), reversed(shape)):
            if idx < 0 or idx >= dim:
                raise IndexError(f"Index {idx} out of bounds for dimension {dim}")
            index += idx * product
            product *= dim
        return index

    _allowed_commands = {'zeros', 'ones', 'rand', 'randn'}

    def __new__(cls, shape: Size, command: str = 'zeros', data: List[float] = None) -> "GenericTensor":
        shape_tuple = tuple(shape)
        total_elements = 1
        for dim in shape_tuple:
            if dim <= 0:
                raise ValueError("Shape dimensions must be positive integers")
            total_elements *= dim

        if data is not None:
            if len(data) != total_elements:
                raise ValueError("Data length does not match the product of the shape dimensions")
            initial_data = data
        else:
            if command not in cls._allowed_commands:
                raise ValueError(f"Invalid command. Allowed values: {cls._allowed_commands}")
            if command == 'zeros':
                initial_data = [0.0] * total_elements
            elif command == 'ones':
                initial_data = [1.0] * total_elements
            elif command == 'rand':
                initial_data = [random.uniform(0, 10) for _ in range(total_elements)]
            elif command == 'randn':
                initial_data = [random.gauss(0, 1) for _ in range(total_elements)]

        obj = super().__new__(cls, 'd', initial_data)
        obj.shape = tuple((1,)) if total_elements == 1 else shape_tuple 
        obj.size=total_elements
        return obj

    def __len__(self):
        total = 1
        for dim in self.shape:
            total *= dim
        return total


    def __add__(self, B):
        if not isinstance(B, GenericTensor):
            return NotImplemented
        if self.shape != B.shape:
            raise ValueError("Shapes must match for addition")
        summed = [A + B for A, B in zip(self, B)]
        return GenericTensor(self.shape, data=summed)

    def __getitem__(self, key: Union[int, Tuple[int, ...]]) -> float:
        if isinstance(key, tuple):
            if len(key) != len(self.shape):
                raise IndexError(f"Expected {len(self.shape)} indices, got {len(key)}")
            linear_index = GenericTensor.__calculate_linear_index(self.shape, key)
            return super().__getitem__(linear_index)
        else:
            return super().__getitem__(key)

    def __setitem__(self, key: Union[int, Tuple[int, ...]], value: float) -> None:
        if isinstance(key, tuple):
            if len(key) != len(self.shape):
                raise IndexError(f"Expected {len(self.shape)} indices, got {len(key)}")
            linear_index = GenericTensor.__calculate_linear_index(self.shape, key)
            super().__setitem__(linear_index, value)
        else:
            super().__setitem__(key, value)

    def __str__(self) -> str:
        def build_string(indices: List[int], depth: int) -> str:
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
        with open(filename, 'w') as f:
            f.write(f"{len(self.shape)}\n")
            f.write(" ".join(map(str, self.shape)) + "\n")
            f.write(" ".join(f"{val:.6f}" for val in self) + "\n")

    @classmethod
    def load(cls, filename: str) -> "GenericTensor":
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist")
        with open(filename, 'r') as f:
            num_dims = int(f.readline().strip())
            shape = tuple(map(int, f.readline().strip().split()))
            if len(shape) != num_dims:
                raise ValueError("Mismatch in the number of dimensions")
            data = list(map(float, f.readline().strip().split()))
            return cls(shape, data=data)


    def __mul__(self, B: Union[float, int, "GenericTensor"]) -> "GenericTensor":
        """
        Element-wise dot multiplication.
        """
        if isinstance(B, (int, float)):
            result_data = [x * B for x in self]
        elif isinstance(B, GenericTensor):
            if self.shape != B.shape:
                raise ValueError("Shapes must match for element-wise multiplication")
            result_data = [x * y for x, y in zip(self, B)]
        else:
            raise TypeError("Unsupported operand type(s) for *")
        return GenericTensor(self.shape, data=result_data)
    
    @classmethod
    def __reshape__(cls , T: "GenericTensor"=None,) -> "GenericTensor":
        if T.size == 1:
            if T is cls:
                s = (1, T.shape[0]) # raw
            else: 
                s = (T.shape[0], 1) # column
            return GenericTensor(s, data=list(T))
        else:
            return T

    def __matmul__(self, B: "GenericTensor") -> "GenericTensor":
        A_shape = self.shape
        B_shape = B.shape
        A_view = self.__reshape__(self)
        B_view = self.__reshape__(B)
        if A_view.shape[-1] != B_view.shape[0]:
            raise ValueError(f"Shapes not aligned for matrix multiplication: {A_shape} and {B_shape}")
        output_shape = A_view.shape[:-1] + B_view.shape[1:]
        
        code_lines = []
        indent = ""
        loop_indices = []
        total_loops = len(output_shape)
        for idx, dim in enumerate(output_shape):
            var = f"i{idx}"
            loop_indices.append(var)
            code_lines.append(f"{indent}for {var} in range({dim}):")
            indent += "    "
          
        sum_lines = []
        sum_lines.append(f"{indent}s = 0.0")
        k_dim = A_view.shape[-1]  # Shared dimension for summation
        sum_lines.append(f"{indent}for k in range({k_dim}):")
        
        a_idx = [f"i{i}" for i in range(len(A_view.shape)-1)] + ["k"]
        b_idx = ["k"] + [f"i{i}" for i in range(len(A_view.shape)-1, total_loops)]
        sum_lines.append(f"{indent}    s += A_view[{', '.join(a_idx)}] * B_view[{', '.join(b_idx)}]")
        sum_lines.append(f"{indent}result.append(s)")
        code_lines.extend(sum_lines)
        exec_code = "\n".join(code_lines)
        local_vars = {"A_view": A_view, "B_view": B_view, "result": []}
        exec(exec_code, {}, local_vars)

        if len(A_shape) == 1 and len(B_shape) == 1:
            return GenericTensor((1,), data=local_vars["result"])
        
        return GenericTensor(output_shape, data=local_vars["result"])

class Tensor(GenericTensor):
    def __new__(cls, shape: Size, command: str = 'zeros', data: List[float] = None) -> "Tensor":
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
        result = super().__matmul__(B)
        return Tensor(result.shape, data=list(result))


class Matrix(Tensor):
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
        result = super().__matmul__(B)
        return Matrix(result.shape, data=list(result))

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    def t(self) -> "Matrix":
        """Transpose"""
        rows, cols = self.shape
        data_t = [self[i, j] for j in range(cols) for i in range(rows)]
        return Matrix((cols, rows), data=data_t)

class Vector(Tensor):
    def __new__(cls, shape: Size=None, command: str = 'zeros', data: List[float] = None ) -> "Vector":
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

    def __matmul__(self, B: "Vector") -> 'Vector':
        result = super().__matmul__(B)
        return Vector(result.shape, data=list(result))

    def norm(self, taste: str = 'euclidean') -> float:
        if taste == 'euclidean':
            return math.sqrt(sum(v**2 for v in self))
        elif taste == 'abs':
            return sum(abs(v) for v in self)
        else:
            raise ValueError("Not permitted")



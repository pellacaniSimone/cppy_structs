"""
test.py

Author: Simone Pellacani (pellacanisimone2017@gmail.com)
Brief: This module provides functionality for testing binary tree operations and performance.
Version: 0.1
Date: 2024-11-08

Copyright (c) 2024

This module implements test cases to validate the functionality and structure of a binary tree (BTree) through various operations, including insertion, deletion, rebalancing, and search. Decorators are used to measure function execution time, and tests ensure structural integrity, balance, and correct node handling.

Functions:
    - Decorators.timeExecution: A decorator that tracks the execution time of functions.
    - TestBTree.test_sequential_insertions: Tests sequential insertion into the tree and checks depth and balance.
    - TestBTree.test_rebalance_after_inserts: Verifies rebalancing after numerous ordered insertions.
    - TestBTree.test_delete_nodes_and_rebalance: Tests node deletion and rebalancing.
    - TestBTree.test_search_non_existent: Searches for a non-existent node and verifies the result.
    - TestBTree.test_invalid_insert_type: Checks that invalid types raise an error on insertion.
    - TestBTree.test_balanced_structure_after_deletes: Tests rebalancing after deleting multiple nodes.
    - TestBTree.test_string_representation: Verifies the textual representation of the tree structure.

Example:
    This module can be executed to perform all tests:
    ```
    python test.py
    ```
"""

from btnode import BNode
from btree import BTree
report = []
import time

class Decorators:
    
    @staticmethod
    def timeExecution(func):
        """Decorator to record start and end timestamps of the function execution in the report."""
        global report
        report = []
        def wrapper(*args, **kwargs):
            report.append(str(time.time()) + "\n")
            result = func(*args, **kwargs)
            report.append(str(time.time()) + "\n")
            return result
        return wrapper

import unittest

class TestBTree(unittest.TestCase):

    def test_sequential_insertions(self):
        """Tests sequential insertion of values in increasing order, verifying depth and balance."""
        tree = BTree()
        for i in range(1, 16):  # Test with 15 nodes
            tree.insert(i)
        self.assertEqual(tree._size, 15)
        self.assertLessEqual(tree.calculateHeight(), 4)

    def test_rebalance_after_inserts(self):
        """Tests tree rebalancing after many sequential insertions."""
        tree = BTree()
        for i in range(1, 50):  
            tree.insert(i)
        tree.rebalance()
        self.assertLessEqual(tree.calculateHeight(), 6)

    def test_delete_nodes_and_rebalance(self):
        """Tests node deletion and rebalancing of the tree."""
        tree = BTree()
        for i in range(1, 11):  
            tree.insert(i)
        tree.delete(5)  
        self.assertIsNone(tree.search(5))  
        self.assertEqual(tree._size, 9)  
        tree.rebalance()
        self.assertLessEqual(tree.calculateHeight(), 4)

    def test_search_non_existent(self):
        """Searches for a non-existent node, verifying the result is None."""
        tree = BTree()
        for i in range(1, 21):  
            tree.insert(i)
        self.assertIsNone(tree.search(100)) 

    def test_invalid_insert_type(self):
        """Tests insertion of an incompatible type."""
        tree = BTree(10) 
        with self.assertRaises(TypeError):
            tree.insert("string")  

    def test_balanced_structure_after_deletes(self):
        """Deletes multiple nodes and verifies rebalancing of the structure."""
        tree = BTree()
        for i in range(1, 100):  
            tree.insert(i)
        for i in range(1, 100, 10): 
            tree.delete(i)
        tree.rebalance()  
        self.assertLessEqual(tree.calculateHeight(), 7)

    def test_string_representation(self):
        """Verifies the text representation of a complex tree structure."""
        tree = BTree(1)
        values = [2, 3, 4, 5, 6, 7]
        for v in values:
            tree.insert(v)
        expected_str = "(1(2(4)(5))(3(6)(7)))"  
        self.assertEqual(str(tree), expected_str)

if __name__ == "__main__":
    unittest.main()

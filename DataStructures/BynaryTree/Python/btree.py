"""
btree.py

Author: Simone Pellacani (pellacanisimone2017@gmail.com)
Brief: A module implementing a binary tree with insertion, deletion, search, and balancing functionalities.
Version: 0.1
Date: 2024-11-08

Copyright (c) 2024

This module provides an implementation for a binary tree structure with various functionalities, including node insertion, in-order traversal, balancing, and deletion. Each function is designed to operate on nodes and ensure efficient handling of binary tree properties.

Classes:
    - BTreeMeta: A meta-class that enhances the BTree with mathematical utilities, such as log base 2 calculations and height estimation.
    - BTree: A binary tree structure that supports insertion, deletion, traversal, rebalancing, and searching for elements.

Functions:
    - insert(value): Inserts a node with the specified value.
    - delete(key): Deletes a node with the specified key and rebalances the tree if necessary.
    - search(key): Searches for a node containing the specified key.
    - rebalance(): Rebalances the tree to ensure optimal height.
    - __inOrderTraversal(): Returns the tree elements in in-order traversal.

Example:
    ```python
    # Example of how to use the BTree class
    tree = BTree()
    tree.insert(10)
    tree.insert(5)
    tree.insert(15)
    tree.rebalance()
    node = tree.search(5)
    print(node)  # Prints the node with value 5 if found
    tree.delete(10)
    ```

"""

from btnode import BNode

class BTreeMeta(type):
    """Mathematical Meta-Class for binary tree utilities such as height calculations."""
    
    def __new__(cls, name, bases, class_dict):
        def logBase2(self, value: int) -> int:
            """Calculates the integer log base 2 of a value."""
            log = 1
            if value <= 0:
                return 0
            while value > 1:
                value >>= 1
                log += 1
            return log

        class_dict['logBase2'] = logBase2
        class_dict['calculateHeight'] = lambda self: int(self.logBase2(self._size + 1)) - 1
        return super().__new__(cls, name, bases, class_dict)

class BTree(metaclass=BTreeMeta):
    """Binary Tree class supporting insertion, deletion, search, and balancing."""
    
    def __resetInsertionQueue(self):
        """Initializes the insertion queue for faster insertion handling."""
        self.__insertionQueue = [self.root] if self.root else []

    def __incrementNodeCount(self):
        """Increments node count and recalculates tree height."""
        self._size += 1
        self.__height = self.calculateHeight()

    def __initRoot(self, rootValue: int):
        """Initializes the root node with the specified value."""
        self.root = BNode(rootValue) if bool(rootValue) else None
        self.__type = type(rootValue) if bool(rootValue) else None  # Compatibility tag
        self.__resetInsertionQueue()
        self._size = 0
        self.__height = 0
        if bool(rootValue):
            self.__incrementNodeCount()

    def __init__(self, rootValue=None):
        """Constructor for BTree, optionally initializing with a root value."""
        self.__initRoot(rootValue)

    def __eq__(self, b1: BNode, b2: BNode):
        """Checks equality between two nodes."""
        return b1 == b2

    def __str__(self):
        """Returns a string representation of the tree."""
        def buildString(node):
            if node is None:
                return ""
            leftStr = buildString(node.getLeft())
            rightStr = buildString(node.getRight())
            return f"({node}{leftStr}{rightStr})"
        return buildString(self.root) if self.root else ""

    def __updateQueue(self, current):
        """Adds both child nodes of the current node to the insertion queue if present."""
        self.__insertionQueue.extend(current.getBoth())

    def __setFirstFree(self, current, newNode):
        """Inserts a new node as a child of the current node based on availability."""
        if current.hasOneChild():
            newNode.setParent(current)
            current.setRight(newNode)
            self.__updateQueue(current)
            return True
        elif current.isLeaf():
            newNode.setParent(current)
            current.setLeft(newNode)
            self.__insertionQueue.insert(0, current)
            return True
        else:
            raise Exception('Unexpected error in node placement')

    def insert(self, value):
        """Inserts a node with the specified value into the tree."""
        if self.__type and not isinstance(value, self.__type):
            raise TypeError(f"Incompatible value type: expected {self.__type.__name__}, got {type(value).__name__}")
        if bool(value):
            if self.root is None:
                self.__initRoot(value)
            else:
                newNode = BNode(value)
                while self.__insertionQueue and newNode:
                    current = self.__insertionQueue.pop(0)
                    if current.hasBothChildren():
                        self.__updateQueue(current)
                    elif self.__setFirstFree(current, newNode):
                        newNode = None
                self.__incrementNodeCount()

    def __inOrderTraversal(self):
        """Performs an in-order traversal of the tree and returns elements in a list."""
        def traverse(node):
            return traverse(node.getLeft()) + [node.getInfo()] + traverse(node.getRight()) if node else []
        return traverse(self.root)

    def __insertRebalance(self):
        """Rebuilds the tree to be balanced based on in-order traversal elements."""
        elements = self.__inOrderTraversal()
        self.root = None
        for elem in elements:
            self.insert(elem)

    def rebalance(self):
        """Public method to rebalance the tree."""
        self.__insertRebalance()

    def search(self, key):
        """Searches for a node with the specified key in the tree."""
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node is None:
                continue
            if node.getInfo() == key:
                return node
            queue.extend(node.getBoth())
        return None

    def delete(self, key):
        """Deletes the specified key from the tree and rebalances."""
        elements = [x for x in self.__inOrderTraversal() if x != key]
        self._size = len(elements)
        self.__height = self.calculateHeight()
        self.__initRoot(None)
        for elem in elements:
            self.insert(elem)

"""
btnode.py

Author: Simone Pellacani (pellacanisimone2017@gmail.com)
Summary: This module defines the BNode class, which represents nodes in a binary tree.
Version: 0.1
Date: 2024-11-08

Copyright (c) 2024

This module provides an implementation of a basic binary tree node with methods for setting and accessing node information, 
checking relationships (parent, left, and right nodes), and comparing nodes. Nodes can be iterated over in a tree traversal 
and support hashing for use in data structures like sets and dictionaries.

Classes:
    - BNode: Implements a binary tree node with information, left, right, and parent references.

Example:
    Example usage of BNode:
    ```
    root = BNode(10)
    left_child = BNode(5)
    right_child = BNode(15)
    root.setLeft(left_child)
    root.setRight(right_child)
    ```

"""

class BNode:
    def __init__(self, inf: int):
        self.__inf = inf
        self.__left = None
        self.__right = None
        self.__parent = None
        self.__hash_value = None

    def __iter__(self):
        yield self

    def __next__(self):
        if self.__left:
            yield from self.__left
        if self.__right:
            yield from self.__right

    def __str__(self) -> str:
        return str(self.__inf)

    def __eq__(self, other) -> bool:
        return (self.__inf == other.getInfo() and
                self.__left == other.getLeft() and
                self.__right == other.getRight()) if other else False

    def __lt__(self, other) -> bool:
        return self.__inf < other.getInfo()

    def __le__(self, other) -> bool:
        return self.__inf <= other.getInfo()

    def __gt__(self, other) -> bool:
        return self.__inf > other.getInfo()

    def __ge__(self, other) -> bool:
        return self.__inf >= other.getInfo()

    def __ne__(self, other) -> bool:
        return self.__inf != other.getInfo()

    def __computeHash(self):
        hl = hash(self.__left) if self.__left else 0
        hr = hash(self.__right) if self.__right else 0
        self.__hash_value = hash((self.__inf, hl, hr))
        return self.__hash_value

    def __hash__(self):
        return self.__hash_value if self.__hash_value else self.__computeHash()

    def isLeaf(self) -> bool:
        return not (self.__left or self.__right)

    def hasOneChild(self) -> bool:  # Logical XOR
        return bool(self.__right) ^ bool(self.__left)

    def hasBothChildren(self) -> bool:
        return bool(self.__right and self.__left)

    def setInfo(self, val: int):
        self.__inf = val

    def setLeft(self, left: 'BNode'):
        self.__left = left

    def setRight(self, right: 'BNode'):
        self.__right = right

    def setParent(self, par: 'BNode'):
        self.__parent = par

    def getLeft(self) -> 'BNode':
        return self.__left

    def getRight(self) -> 'BNode':
        return self.__right

    def getParent(self) -> 'BNode':
        return self.__parent

    def getInfo(self) -> int:
        return self.__inf

    def getBoth(self):
        return [self.__left, self.__right]

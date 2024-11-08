/**
 * @file btree.h
 * @author Simone Pellacani (pellacanisimone2017@gmail.com)
 * @brief This module provides an implementation for a binary tree structure with various 
 * functionalities, including node insertion, in-order traversal, balancing, and deletion. 
 * Each function is designed to operate on nodes and ensure efficient 
 * handling of binary tree properties.
 * 
 * @version 0.1
 * @date 2024-11-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef BTREE_HPP
#define BTREE_HPP


#include <vector> // for delete
#include <cmath>
#include <deque>
#include <iostream>
#include <memory>


#include "bnode.h"
using namespace std;
class BTree {
private:
    shared_ptr<BNode> root;
    deque<shared_ptr<BNode>> insertionQueue;
    int size;
    int height;

    static int calculateHeight(int size);
    void incrementNodeCount();
    bool setFirstFree(const shared_ptr<BNode>& current, const shared_ptr<BNode>& newNode);
    static void printPreOrderRecursive(const shared_ptr<BNode>& node);
    static bool sameRecursive(const shared_ptr<BNode>& b1, const shared_ptr<BNode>& b2);
    void insertRoot(const shared_ptr<BNode>& n);
    shared_ptr<BNode> searchNode(const shared_ptr<BNode>& node, const TipoInf& key) const;
    void inOrderTraversal(const shared_ptr<BNode>& node, vector<TipoInf>& values);
    void createTreeFromValues(const vector<TipoInf>& values, bool toDelete = false, const TipoInf& delKey = TipoInf());
    void deleteNodeRecursive(const shared_ptr<BNode>& node);

public:
    BTree(const shared_ptr<BNode>& n = nullptr);
    int getSize() const;
    int getHeight() const;
    bool same(const BTree& other) const;
    void insert(TipoInf value);
    void fullTreeDelete();
    void printPreOrder() const;
    shared_ptr<BNode> search(const TipoInf& key) const;
    void rebalance();
    void deleteNode(const TipoInf& key);
};


#endif

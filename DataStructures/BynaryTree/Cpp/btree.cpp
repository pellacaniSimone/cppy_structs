/**
 * @file btree.cpp
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

#include "btree.h"

#include <cmath>
#include <iostream>

int BTree::calculateHeight(int size) {
    return log2(size + 1);
}

void BTree::incrementNodeCount() {
    size++;
    height = calculateHeight(size);
}

bool BTree::setFirstFree(const shared_ptr<BNode>& current, const shared_ptr<BNode>& newNode) {
    if (current->hasOneChild()) {
        newNode->setParent(current);
        current->setRight(newNode);
        insertionQueue.push_front(current->getLeft());
        insertionQueue.push_front(current->getRight());
        return true;
    }
    if (current->isLeaf()) {
        newNode->setParent(current);
        current->setLeft(newNode);
        insertionQueue.push_back(current);
        return true;
    }
    return false;
}

void BTree::printPreOrderRecursive(const shared_ptr<BNode>& node) {
    if (!node) return;
    cout << "(";
    Tipo::print(node->getInf());
    printPreOrderRecursive(node->getLeft());
    printPreOrderRecursive(node->getRight());
    cout << ")";
}

bool BTree::sameRecursive(const shared_ptr<BNode>& b1, const shared_ptr<BNode>& b2) {
    if (!b1 || !b2) return b1 == b2;
    return (b1->getInf() == b2->getInf()) &&
           sameRecursive(b1->getLeft(), b2->getLeft()) &&
           sameRecursive(b1->getRight(), b2->getRight());
}

void BTree::insertRoot(const shared_ptr<BNode>& n) {
    if (root) cerr << "Error: root already exists.\n";
    else root = n;
}

shared_ptr<BNode> BTree::searchNode(const shared_ptr<BNode>& node, const TipoInf& key) const {
    if (!node) return nullptr;
    if (node->getInf() == key) return node;

    auto leftResult = searchNode(node->getLeft(), key);
    if (leftResult) return leftResult;

    return searchNode(node->getRight(), key);
}

void BTree::inOrderTraversal(const shared_ptr<BNode>& node, vector<TipoInf>& values) {
    if (!node) return;
    if (node->getLeft())
        inOrderTraversal(node->getLeft(), values);
    values.push_back(node->getInf());
    if (node->getRight())
        inOrderTraversal(node->getRight(), values);
}

void BTree::createTreeFromValues(const vector<TipoInf>& values, bool toDelete, const TipoInf& delKey) {
    for (const auto& value : values) {
        if (toDelete && value == delKey) {
            toDelete = false;
            continue;
        }
        insert(value);
    }
}

void BTree::deleteNodeRecursive(const shared_ptr<BNode>& node) {
    if (!node) return;
    if (node->getLeft())
        deleteNodeRecursive(node->getLeft());
    if (node->getRight())
        deleteNodeRecursive(node->getRight());
    node->setLeft(nullptr);  
    node->setRight(nullptr); 
    node->setParent(nullptr);
}

BTree::BTree(const shared_ptr<BNode>& n) : root(n), size(n ? 1 : 0), height(calculateHeight(size)) {
    if (n) insertionQueue.push_front(n);
}

int BTree::getSize() const { return size; }

int BTree::getHeight() const { return height; }

bool BTree::same(const BTree& other) const { return sameRecursive(root, other.root); }

void BTree::insert(TipoInf value) {
    auto newNode = make_shared<BNode>(value);
    if (!root) {
        insertRoot(newNode);
        incrementNodeCount();
        insertionQueue.push_front(newNode);
    } else {
        bool inserted = false;
        while (!inserted) {
            auto current = insertionQueue.front();
            insertionQueue.pop_back();
            if (current->hasBothChildren()) {
                insertionQueue.push_front(current->getLeft());
                insertionQueue.push_front(current->getRight());
            }
            if (setFirstFree(current, newNode)) {
                inserted = true;
                incrementNodeCount();
            }
        }
    }
}

void BTree::fullTreeDelete() {
    deleteNodeRecursive(root);
    root.reset();  
    insertionQueue.clear();  
    size = 0;
    height = 0;
}

void BTree::printPreOrder() const {
    if (!root) { cout << "Tree is empty.\n"; return; }
    printPreOrderRecursive(root);
    cout << '\n';
}

shared_ptr<BNode> BTree::search(const TipoInf& key) const {
    return searchNode(root, key);
}

void BTree::rebalance() {
    if (!root) return;
    vector<TipoInf> values;
    inOrderTraversal(root, values);
    fullTreeDelete();
    createTreeFromValues(values, false, 0);
}

void BTree::deleteNode(const TipoInf& key) {
    auto nodeToDelete = search(key);
    if (!nodeToDelete) return;
    vector<TipoInf> values;
    inOrderTraversal(root, values);
    fullTreeDelete();
    createTreeFromValues(values, true, key);
}

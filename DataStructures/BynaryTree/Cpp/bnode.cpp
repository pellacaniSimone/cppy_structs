/**
 * @file bnode.cpp
 * @author Simone Pellacani (pellacanisimone2017@gmail.com)
 * @brief This module provides an implementation of a basic binary tree 
 * node with methods for setting and accessing node information, 
 * checking relationships (parent, left, and right nodes), 
 * and comparing nodes. 
 * 
 * @version 0.1
 * @date 2024-11-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#include "bnode.h"


// construct
BNode::BNode(TipoInf i) : inf(i), parent(std::weak_ptr<BNode>()), left(nullptr), right(nullptr) {}
BNode::~BNode() {;}


// getters
TipoInf BNode::getInf() const { return inf; }
std::weak_ptr<BNode> BNode::getParent() const { return parent; }
std::shared_ptr<BNode> BNode::getLeft() const { return left; }
std::shared_ptr<BNode> BNode::getRight() const { return right; }

// setters
void BNode::setLeft(const std::shared_ptr<BNode>& t) { left = t; }
void BNode::setRight(const std::shared_ptr<BNode>& t) { right = t; }
void BNode::setParent(const std::shared_ptr<BNode>& t) { parent = t; }

// getSomeState
bool BNode::isLeaf() const { return countChildren() == 0; }
bool BNode::hasOneChild() const { return countChildren() == 1; }
bool BNode::hasBothChildren() const { return countChildren() == 2; }

// private
int BNode::countChildren() const { return bool(left) + bool(right); }


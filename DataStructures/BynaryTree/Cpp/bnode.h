/**
 * @file bnode.h
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

#ifndef BNODE_HPP
#define BNODE_HPP

#include "tipo.h"

#include <memory> // shared_ptr  weak_ptr

class BNode {
private:

    // data incapsulation
    TipoInf inf;
    std::weak_ptr<BNode> parent;
    std::shared_ptr<BNode> left;
    std::shared_ptr<BNode> right;
    int countChildren() const ;

public:

    // de/con ~ struct
    BNode(TipoInf i);
    ~BNode();

    // getters
    TipoInf getInf() const;
    std::weak_ptr<BNode> getParent() const;
    std::shared_ptr<BNode> getLeft() const;
    std::shared_ptr<BNode> getRight() const;

    // setters
    void setLeft(const std::shared_ptr<BNode>& t);
    void setRight(const std::shared_ptr<BNode>& t);
    void setParent(const std::shared_ptr<BNode>& t);

    // getSomeState
    bool isLeaf() const;
    bool hasOneChild() const;
    bool hasBothChildren() const;
};

#endif

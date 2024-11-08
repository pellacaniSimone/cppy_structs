/**
 * @file bnode.c
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

bnode* getNewBTNode(tipo_inf i) {
  bnode* r = (bnode*) malloc(sizeof(bnode));
  r->inf =i ;
  r->left = r->right = r->parent = NULL;
  return r;
}

void setLeft( struct bnode* p, struct bnode* t) { p->left = t; }
void setRight(struct bnode* p, struct bnode* t) { p->right = t; }
void setParent(struct bnode* p, struct bnode* t) { p->parent = t; }


struct bnode* getLeft( struct bnode* node) { return node->left; }
struct bnode* getRight(struct bnode* node) { return node->right; }
struct bnode* getParent(struct bnode* node) { return node->parent; }
tipo_inf getInfo(struct bnode* node) { return node->inf; }



static int countChildren(bnode* node) { return (node->left != NULL) + (node->right != NULL); }
bool isLeaf(struct bnode* node) {return countChildren(node) == 0;}
bool hasOneChild(bnode* node) { return countChildren(node) == 1; }
bool hasBothChildren(bnode* node) { return countChildren(node) == 2; }
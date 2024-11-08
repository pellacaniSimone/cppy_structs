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

#ifndef BTREE_STRUCT
#define BTREE_STRUCT

#include "bnode.h"
#include "deque.h"

typedef struct {
  struct bnode *root;
  coda *insertionQueue;
  int size;
  int height;
} btree;


//enum bool { false, true } ;
#include <stdbool.h>

btree* newTree(bnode*);
bnode* getRoot(btree*);
void insertRoot(btree*,bnode*);


bool same(btree *, btree * );
void insert(btree* , tipo_inf );
void freeTree(bnode * );
void fullTreeDelete(btree * );
bnode* search(btree* , tipo_inf );
btree * rebalance(btree *);
btree * deleteNode(btree* , tipo_inf );
void printPreOrder(btree *);

#endif 
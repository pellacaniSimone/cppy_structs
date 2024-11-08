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

#ifndef BNODE_STRUCT
#define BNODE_STRUCT

#include <stdio.h> // #include <iostream>
#include <stdlib.h>
#include <stdbool.h>


#include "tipo.h"

typedef struct bnode {
  tipo_inf inf;
  struct bnode* parent;
  struct bnode* left;
  struct bnode* right;
} bnode;


struct bnode* getNewBTNode(tipo_inf);

void setLeft(struct bnode*, struct bnode*);
void setRight(struct bnode*, struct bnode*);
void setParent(struct bnode*, struct bnode*);
//void set_info(struct bnode*,tipo_inf );

struct bnode* getLeft(struct bnode*);
struct bnode* getRight(struct bnode*);
struct bnode* getParent(struct bnode*);
tipo_inf getInfo(struct bnode*);


bool isLeaf (struct bnode* );
bool hasOneChild (struct bnode* );
bool hasBothChildren (struct bnode* );

#endif 
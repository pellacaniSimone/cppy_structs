/**
 * @file btree.c
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


#include <stdio.h> // #include <iostream>
#include "btree.h"
#include "deque.h"



static int logBase2(int value) {
  int log = 1;
  if (value <= 0) { return 0; }
  while (value > 1) {
    value >>= 1;  
    log += 1;
  }
  return log;
}

static int calculateHeight(int size) { return logBase2(size + 1) - 1; }

static void incrementNodeCount(btree *t) {
  t->size++;
  t->height = calculateHeight(t->size);
}

btree* newTree(bnode* n) {
  btree* r = (btree*)malloc(sizeof(btree));
  r->root = n;
  r->size = (n == NULL) ? 0 : 1;
  r->height = calculateHeight(r->size);
  r->insertionQueue = newQueue();
  if (n != NULL) enqueue(r->insertionQueue, n);
  return r;
}


static bool same_recoursive(bnode * b1, bnode * b2) {
  if (b1 == NULL || b2 == NULL) 
    return (b1 == NULL && b2 == NULL);
  else {
    bool left = same_recoursive(b1->left, b2->left);
    bool right = same_recoursive(b1->right, b2->right);
    return b1->inf == b2->inf && left && right;
  }
}

bnode* getRoot(btree *t) { return t->root; }

void insertRoot(btree *t, bnode* n) {
  if (getRoot(t) != NULL) 
    printf("Error: root already exists.\n");
  else
    t->root = n;
}

bool same(btree * b1, btree * b2) {
  return same_recoursive(getRoot(b1) , getRoot(b2)  );
}



static bool setFirstFree(btree *t, bnode *current, bnode *newNode) {
  if ( hasOneChild(current)) { // empty right child
    newNode->parent = current;
    current->right = newNode;
    enqueue(t->insertionQueue,getLeft(current));
    enqueue(t->insertionQueue,getRight(current));
    return true;
  } 
  else if (isLeaf(current) ) { // empty left child
    newNode->parent = current;
    current->left = newNode;
    push(t->insertionQueue, current);
    return true;
  }
  return false;
}


void insert(btree *t, tipo_inf value) {
  bnode* newNode = getNewBTNode(value);

  if (getRoot(t) == NULL) {
    insertRoot(t, newNode);
    incrementNodeCount(t);
    enqueue(t->insertionQueue, newNode);
  } 
  else {
    bool inserted = false;
    while (!inserted) {
      bnode* current = dequeue(t->insertionQueue);
      if (hasBothChildren(current)) { // both not NULL
        enqueue(t->insertionQueue,getLeft(current));
        enqueue(t->insertionQueue,getRight(current));
      }
      if (setFirstFree(t, current, newNode)) {
        inserted = true;
        incrementNodeCount(t);
      } 
    }
  }
}



void freeTree(bnode *node) {
  if (node != NULL) {
    freeTree(node->left);
    freeTree(node->right);
    free(node);  
  }
}

void fullTreeDelete(btree * t) {
  freeQueue(t->insertionQueue);
  freeTree(t->root) ;
  free(t);
}

static bnode* searchNode(bnode* n, tipo_inf key) {
  if (n == NULL) return NULL;
  if (compare(getInfo(n), key) == 0) return n;

  bnode* left_result = searchNode(getLeft(n), key);
  if (left_result != NULL) return left_result;
  
  return searchNode(getRight(n), key);
}

bnode* search(btree * t, tipo_inf key) { return searchNode(getRoot(t) , key);}

static void inOrderTraversal(bnode* node, tipo_inf * arr, int * index) {
  if (node == NULL) return;
  if (node->left  != NULL)
    inOrderTraversal(node->left,  arr,  index );
  arr[index[0]++] = node->inf;
  if (node->right != NULL)
    inOrderTraversal(node->right, arr,  index );
}


static btree* createTreeFromValues(tipo_inf* values, int * size, bool toDelete, tipo_inf delKey) {
  btree* tree = NULL;
  bool first = true;
  for (int i = 0; i < size[0]; i++) {
    if  ( toDelete && compare(values[i], delKey) == 0 ) {
      toDelete=false; continue;
    }
    if (first) {
      tree = newTree(getNewBTNode(values[i]));
      first = false;
    } else {
      insert(tree, values[i]);
    }
  }
  return tree;
}


btree * rebalance(btree * tree){
  if (tree == NULL || getRoot(tree) == NULL) return NULL;
  tipo_inf* values = (tipo_inf*) malloc(tree->size * sizeof(tipo_inf));
  int index[] = {0};
  inOrderTraversal(tree->root, values, index);
  fullTreeDelete(tree);
  tipo_inf app=0;
  tree = createTreeFromValues(values,index,false,app); // false app, not delete
  free(values);
  return tree;
}


btree * deleteNode(btree *tree, tipo_inf key) {
  if (tree == NULL || getRoot(tree) == NULL) return false;
  bnode *to_delete = search(tree, key);
  if (to_delete == NULL ) return false;
  tipo_inf* values = (tipo_inf*) malloc(tree->size * sizeof(tipo_inf));
  int index[] = {0};
  inOrderTraversal(tree->root, values, index);
  fullTreeDelete(tree);
  tree = createTreeFromValues(values,index,true,key); // true delete
  free(values);
  return tree;
}

static void  printPreOrderRecursive(bnode *node) {
  if (node == NULL) { return; }
  else {
    printf("(");  
    if (getLeft(node) !=  NULL)
      printPreOrderRecursive(getLeft(node)); 
    print(&node->inf);  
    if (getRight(node) !=  NULL )
      printPreOrderRecursive(getRight(node)); 
    printf(")");
  }
}
void printPreOrder(btree *tree) {
  if (tree == NULL || tree->root == NULL) {
    printf("L'albero è vuoto.\n");
    return;
  }
  printPreOrderRecursive(tree->root);
  printf("\n");  
} 

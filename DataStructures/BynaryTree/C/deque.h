/**
 * @file deque.h
 * @author Simone Pellacani (pellacanisimone2017@gmail.com)
 * @brief Not indexed sequence container that allows fast insertion 
 * and deletion at both its beginning and its end.
 * 
 * @version 0.1
 * @date 2024-11-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef BFS_T_QUEUE
#define BFS_T_QUEUE

#include <stdlib.h>

#include "bnode.h"

typedef bnode* tnode_inf;

typedef struct bfsn {
  tnode_inf inf;
  struct bfsn* next;
} bfsn;

typedef bfsn* slist;


typedef struct {
  slist head;
  bfsn* tail;
} coda;

coda* newQueue();
void push(coda * , tnode_inf );
void  enqueue(coda* , tnode_inf );
tnode_inf dequeue(coda* );
void freeQueue(coda* q);


#endif
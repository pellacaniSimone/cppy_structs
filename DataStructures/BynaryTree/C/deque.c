/**
 * @file bfsqueue.c
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
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "deque.h"



typedef bnode* tnode_inf;

// struct list

typedef bfsn* slist;

static slist newElem(tnode_inf n) {
  slist r = (slist)malloc(sizeof(bfsn));
  r->inf = n;
  r->next = NULL;
  return r;
}

// struct coda

static bool isEmptyQ(coda* q) { return (q == NULL || q->head == NULL); }

coda* newQueue() {
  coda* q = (coda*)malloc(sizeof(coda));
  q->head = NULL;
  q->tail = NULL;
  return q;
}


void push(coda * q, tnode_inf i) {
  if (i == NULL) return ;             
  if (q == NULL) q = newQueue();       
  bfsn *e = newElem(i);               
  if (isEmptyQ(q)) {
    q->head = e;
    q->tail = e;
  } else {
    e->next = q->head;
    q->head = e; 
  }
}



void  enqueue(coda* q, tnode_inf i) {
  if (q == NULL) q = newQueue();
  bfsn* e = newElem(i);
  if (isEmptyQ(q)) {
    q->head = e;
    q->tail = e;
  } else {
    q->tail->next = e;
    q->tail = e;
  }
}

tnode_inf dequeue(coda* q) {
  if (isEmptyQ(q)) return NULL; //  isEmptyQ
  slist node_to_remove = q->head;
  tnode_inf result = node_to_remove->inf;
  q->head = q->head->next;
  if (q->head == NULL) {
    q->tail = NULL; // reset
  }
  free(node_to_remove);
  return result;
}


void freeQueue(coda* q) {
  while (!isEmptyQ(q)) {
    dequeue(q);
  }
  free(q);
}


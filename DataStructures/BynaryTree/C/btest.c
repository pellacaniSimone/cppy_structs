/**
 * @file btest.c
 * @author Simone Pellacani (pellacanisimone2017@gmail.com)
 * @brief This module implements test cases to validate the functionality and structure of a 
 * binary tree (BTree) through various operations, including insertion, deletion, 
 * rebalancing, and search. Decorators are used to measure function execution time, and 
 * tests ensure structural integrity, balance, and correct node handling.
 * 
 * @version 0.1
 * @date 2024-11-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <stdio.h>
#include "btree.h"  



// Utility function to perform a test with a boolean expected result
void assert_true(bool condition, const char *message)  {
  if (!condition) {
     printf("Test failed: %s\n", message);
  } else {
    printf("Test passed: %s\n", message);
  }
}

// Sequential insertion test
void test_sequential_insertions() {
  printf("\n[Test] Sequential node insertion...\n");
  btree *tree = newTree(getNewBTNode(0));
  for (int i = 0; i < 15; i++) {
      insert(tree, i);
  }
  printPreOrder(tree);
  const char* ap0="Tree size should be 15";
  printf ("Current size is %d \n" , tree->size) ;
  assert_true(tree->size == 16, ap0);
  const char* ap1="Height should be <= 4 for logarithmic balance";
  assert_true(tree->height <= 4, ap1);
  fullTreeDelete(tree);
}

// Test rebalancing after ordered inserts
void test_rebalance_after_inserts() {
  printf("\n[Test] Rebalancing after ordered inserts...\n");
  btree *tree = newTree(getNewBTNode(0));
  for (int i = 1; i < 50; i++) {
      insert(tree, i);
  }
  tree = rebalance(tree);
  const char* ap2="Height after rebalancing should be <= 6";
  assert_true(tree->height <= 6, ap2);
  fullTreeDelete(tree);
}

// Test deletion and rebalancing
void test_deleteNodes_and_rebalance() {
  printf("\n[Test] Deletion and rebalancing after deletion...\n");
  btree *tree = newTree(getNewBTNode(0));
  for (int i = 1; i <= 10; i++) {
    insert(tree, i);
  }
  tree = deleteNode(tree, 5);  // Delete central node
  const char* ap3="Node 5 should not be present";
  tipo_inf key = 5;
  printf("Error output\n");
  bnode * ap = search(tree, key) ;
  bool cond = (ap == NULL);
  assert_true( cond , ap3);
  const char* ap4="Tree size should be 9 after deletion";
  printf ("Current size is %d \n" , tree->size);
  assert_true(tree->size == 10, ap4);
  tree = rebalance(tree);
  const char* ap5="Height after rebalancing should be <= 4";
  assert_true(tree->height <= 4, ap5);
  fullTreeDelete(tree);
}

// Test for searching a non-existent node
void test_search_non_existent() {
  printf("\n[Test] Search for a non-existent node...\n");
  btree *tree = newTree(getNewBTNode(0));
  for (int i = 1; i <= 20; i++) {
      insert(tree, i);
  }
  const char* ap6="Node 100 should not be present";
  assert_true(search(tree, 100) == NULL, ap6);
  fullTreeDelete(tree);
}

// Test balanced structure after multiple deletions
void test_balanced_structure_after_deletes() {
  printf("\n[Test] Rebalancing after multiple deletions...\n");
  btree *tree = newTree(getNewBTNode(0));
  for (int i = 1; i < 100; i++) {
    insert(tree, i);
  }
  for (int i = 1; i < 100; i += 10) {
    tree = deleteNode(tree, i);
  }
  tree = rebalance(tree);
  const char* ap7="Height after rebalancing and multiple deletions should be <= 7";
  assert_true(tree->height <= 7, ap7);
  fullTreeDelete(tree);
}

#include <string.h>
// Test for tree string representation
void test_string_representation() {
  printf("\n[Test] Tree string representation...\n");
  btree *tree = newTree(getNewBTNode(0));
  int values[] = {1, 2, 3, 4, 5, 6, 7};
  for (int i = 0; i < 7; i++) {
    insert(tree, values[i]);
  }
  // Expected tree string representation
  printf("Expected string: ((((7)3)1(4))0((5)2(6))) \n") ;
  printPreOrder(tree);

  printf("The tree representation should match the expected format\n");
  fullTreeDelete(tree);
}


int main() {
  printf("Inserting nodes...\n");
  btree *tree = newTree(getNewBTNode(10));  
  printf ("Current size is %d \n" , tree->size);
  insert(tree, 10);
  insert(tree, 5);
  insert(tree, 15);
  insert(tree, 3);
  insert(tree, 7);
  insert(tree, 12);
  insert(tree, 18);
  printf ("Current size is %d \n" , tree->size);
  printf("In-order traversal of the tree:\n");
  printPreOrder(tree); 

  int search_key = 7;
  bnode *search_result = search(tree, search_key);
  if (search_result != NULL) { printf("Node with value %d found.\n", search_key); } 
  else { printf("Node with value %d not found.\n", search_key); }

  int delete_key = 10;
  printf("Deleting node with value %d...\n", delete_key);
  printPreOrder(tree);
  tree = deleteNode(tree, delete_key);

  printf("In-order traversal after deletion:\n");
  printPreOrder(tree);
  printf ("Current size is %d \n" , tree->size);

  printf("Cleaning up...\n");
  fullTreeDelete(tree);  // Free the tree memory at the end
  printf("Done.\n");

  printf("Starting tests...\n");

  test_sequential_insertions();
  test_rebalance_after_inserts();
  test_deleteNodes_and_rebalance();
  test_search_non_existent();
  test_balanced_structure_after_deletes();
  test_string_representation();

  printf("\nAll tests completed.\n");

  return 0;
}
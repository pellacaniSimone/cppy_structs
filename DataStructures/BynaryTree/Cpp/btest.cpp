/**
 * @file btest.cpp
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

#ifndef TEST_BTREE_HPP
#define TEST_BTREE_HPP



#include <iostream>
#include <string>
#include <memory>
#include "btree.h"  

using namespace std;
class TestBTree {
private:

    // write to mantain C compatibility instead of using errors
    void assert_true(bool condition, const string& message) const {
        if (!condition) { cout << "Test failed: " << message << endl; } 
        else { cout << "Test passed: " << message << endl; }
    }


    void testSequentialInsertions() {
        cout << "\n[Test] Sequential node insertions..." << endl;
        BTree tree(make_shared<BNode>(0));
        for (int i = 1; i < 15; i++) { tree.insert(i); }
        tree.printPreOrder();
        assert_true(tree.getSize() == 15, "Tree size should be 15");
        assert_true(tree.getHeight() <= 4, "Height should be <= 4 for logarithmic balance");
        tree.fullTreeDelete();
    }

    // Test rebalance after ordered insertions
    void testRebalanceAfterInserts() {
        cout << "\n[Test] Rebalance after ordered insertions..." << endl;
        BTree tree(make_shared<BNode>(0));
        for (int i = 1; i < 50; i++) { tree.insert(i); }
        tree.rebalance();
        assert_true(tree.getHeight() <= 6, "Height after rebalance should be <= 6");
        tree.fullTreeDelete();
    }

    // Test for deletion and rebalance
    void testDeleteNodesAndRebalance() {
        cout << "\n[Test] Deletion and rebalance after deletion..." << endl;
        BTree tree(make_shared<BNode>(0));
        for (int i = 1; i < 10; i++) { tree.insert(i); }
        cout << "Tree size pre delete: " << tree.getSize() << endl;
        tree.deleteNode(5);  
        assert_true(tree.search(5) == nullptr, "Node 5 should not be present");
        cout << "Tree size after delete: " << tree.getSize() << endl;
        assert_true(tree.getSize() == 9, "Tree size should be 9 after deletion");
        tree.rebalance();
        assert_true(tree.getHeight() <= 4, "Height after rebalance should be <= 4");
        tree.fullTreeDelete();
    }

    // Test searching for a non-existent node
    void testSearchNonExistent() {
        cout << "\n[Test] Search for non-existent node..." << endl;
        BTree tree(make_shared<BNode>(0));
        for (int i = 1; i <= 20; i++) { tree.insert(i); }
        assert_true(tree.search(100) == nullptr, "Node 100 should not be present");
        tree.fullTreeDelete();
    }

    // Test rebalance of structure after multiple deletions
    void testBalancedStructureAfterDeletes() {
        cout << "\n[Test] Rebalance after multiple deletions..." << endl;
        BTree tree(make_shared<BNode>(0));
        for (int i = 1; i < 100; i++) { tree.insert(i);  }
        for (int i = 1; i < 100; i += 10) { tree.deleteNode(i);  }
        tree.rebalance();
        assert_true(tree.getHeight() <= 7, "Height after rebalance and multiple deletions should be <= 7");
        tree.fullTreeDelete();
    }

    // Test tree string representation
    void testStringRepresentation() {
        cout << "\n[Test] Tree string representation..." << endl;
        BTree tree(make_shared<BNode>(0));
        int values[] = {1, 2, 3, 4, 5, 6, 7};
        for (int value : values) { tree.insert(value); }
        cout << "Expected string: (0(1)(2(3)(4(5)(6(7)))))" << endl;
        tree.printPreOrder();
        tree.fullTreeDelete();
    }

    // Test basic insertion
    void testBasicInsertion() {
        cout << "\n[Test] Basic node insertion..." << endl;
        BTree tree(make_shared<BNode>(10));
        tree.insert(5);
        tree.insert(15);
        tree.insert(3);
        tree.insert(7);
        tree.insert(12);
        tree.insert(18);

        cout << "Tree pre-order print:" << endl;
        tree.printPreOrder();
        
        // Check expected size and height
        assert_true(tree.getSize() == 7, "Tree size should be 7");
        assert_true(tree.getHeight() <= 3, "Height should be <= 3 for logarithmic balance");
        tree.fullTreeDelete();
    }

public:
    // Method to run all tests
    void runTests() {
        testSequentialInsertions();
        testRebalanceAfterInserts();
        testDeleteNodesAndRebalance();
        testSearchNonExistent();
        testBalancedStructureAfterDeletes();
        testStringRepresentation();
        testBasicInsertion();
        cout << "\nAll tests completed." << endl;
    }
};

#endif


int main() {
    TestBTree test;
    test.runTests();
    return 0;
}


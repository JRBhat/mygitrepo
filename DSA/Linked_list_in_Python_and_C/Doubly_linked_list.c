"""
    Simple implementation of a doubly linked list in C
"""

#include <stdio.h>
#include <stdlib.h>

// Define a structure for a node in the doubly linked list
struct Node {
    int data;
    struct Node *prev;
    struct Node *next;
};

// Function to create a new node
struct Node* createNode(int data) {
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    if (newNode == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    newNode->data = data;
    newNode->prev = NULL;
    newNode->next = NULL;
    return newNode;
}

int main() {
    // Creating nodes
    struct Node *head = createNode(1);
    struct Node *second = createNode(2);
    struct Node *third = createNode(3);

    // Connecting nodes forward
    head->next = second;
    second->next = third;
    third->next = NULL;

    // Connecting nodes backward
    third->prev = second;
    second->prev = head;
    head->prev = NULL;

    // Traversing and printing the linked list forward
    struct Node *current = head;
    printf("Forward Linked List: ");
    while (current != NULL) {
        printf("%d -> ", current->data);
        current = current->next;
    }
    printf("NULL\n");

    // Traversing and printing the linked list backward
    current = third;
    printf("Backward Linked List: ");
    while (current != NULL) {
        printf("%d -> ", current->data);
        current = current->prev;
    }
    printf("NULL\n");

    // Freeing memory
    free(head);
    free(second);
    free(third);

    return 0;
}

"""
Explanation:
- The `struct Node` represents a node in the doubly linked list. It contains an integer `data`, and two pointers `prev` and `next` to the previous and next nodes in the list respectively.
- The `createNode` function dynamically allocates memory for a new node, initializes its data, and sets its `prev` and `next` pointers to `NULL`.
- In the `main` function, three nodes are created with data values `1`, `2`, and `3`.
- The nodes are connected both forward and backward to form a doubly linked list by setting the `prev` and `next` pointers appropriately.
- The linked list is then traversed and printed both forward and backward.
- Finally, memory allocated for the nodes is freed using the `free` function.
"""
"""Simple implementation of a singly linked list in C:"""

#include <stdio.h>
#include <stdlib.h>

// Define a structure for a node in the linked list
struct Node {
    int data;
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
    newNode->next = NULL;
    return newNode;
}

int main() {
    // Creating nodes
    struct Node *head = createNode(1);
    struct Node *second = createNode(2);
    struct Node *third = createNode(3);

    // Connecting nodes
    head->next = second;
    second->next = third;

    // Traversing and printing the linked list
    struct Node *current = head;
    printf("Linked List: ");
    while (current != NULL) {
        printf("%d -> ", current->data);
        current = current->next;
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
- The `struct Node` represents a node in the linked list. It contains an integer `data` and a pointer `next` to the next node in the list.
- The `createNode` function dynamically allocates memory for a new node, initializes its data, and sets its `next` pointer to `NULL`.
- In the `main` function, three nodes are created with data values `1`, `2`, and `3`.
- The nodes are connected to form a linked list by setting the `next` pointers appropriately.
- The linked list is then traversed starting from the `head` node, and each node's data is printed.
- Finally, memory allocated for the nodes is freed using the `free` function.
"""

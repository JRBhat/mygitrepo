"""Simple implementation of a doubly linked list in Python"""


class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        new_node.prev = current

    def display_forward(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def display_backward(self):
        current = self.head
        while current.next:
            current = current.next
        while current:
            print(current.data, end=" -> ")
            current = current.prev
        print("None")

# Creating a doubly linked list
dllist = DoublyLinkedList()
dllist.append(1)
dllist.append(2)
dllist.append(3)

# Displaying the doubly linked list forward and backward
print("Forward Linked List:")
dllist.display_forward()
print("Backward Linked List:")
dllist.display_backward()


"""
Explanation:
- The `Node` class represents a node in the doubly linked list. It contains a `data` attribute and `prev` and `next` attributes pointing to the previous and next nodes in the list, respectively.
- The `DoublyLinkedList` class represents the doubly linked list itself. It has a `head` attribute pointing to the first node in the list.
- The `append` method adds a new node with the given data to the end of the doubly linked list.
- The `display_forward` method traverses the doubly linked list forward and prints each node's data.
- The `display_backward` method traverses the doubly linked list backward and prints each node's data.
- In the `main` section, a doubly linked list object `dllist` is created, and nodes with data values `1`, `2`, and `3` are appended to it.
- Finally, the doubly linked list is displayed forward and backward using the `display_forward` and `display_backward` methods, respectively.
"""
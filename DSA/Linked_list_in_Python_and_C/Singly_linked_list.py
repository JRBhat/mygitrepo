"""
     Simple implementation of a singly linked list in Python:
"""

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None # initializes the first(head) node to None

    def append(self, data):
        new_node = Node(data) 
        if self.head is None: # if the list is empty
            self.head = new_node #  sets the head to new node
            return
        
        # if list is not empty
        
        last_node = self.head # first sets the temp var "last_node" to the head
        
        while last_node.next: # loops until a node is found which has no next attribute
            last_node = last_node.next # up one node
        last_node.next = new_node # when last node is found, new node is assigned to it

    def display(self):
        current = self.head
        
        # keeps printing the node data until current node is None
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

# Creating a singly linked list
llist = LinkedList()
llist.append(1)
llist.append(2)
llist.append(3)

# Displaying the linked list
print("Linked List:")
llist.display()



"""
Explanation:
- The `Node` class represents a node in the singly linked list. It contains a `data` attribute and a `next` attribute pointing to the next node in the list.
- The `LinkedList` class represents the linked list itself. It has a `head` attribute pointing to the first node in the list.
- The `append` method adds a new node with the given data to the end of the linked list.
- The `display` method traverses the linked list and prints each node's data.
- In the `main` section, a linked list object `llist` is created, and nodes with data values `1`, `2`, and `3` are appended to it.
- Finally, the linked list is displayed using the `display` method.
"""
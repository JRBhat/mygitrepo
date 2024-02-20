"""
Implementation of a simple to-do list application using a singly linked list. 
Each node in the list represents a task, containing the task description and its priority level. 
Provides functionalities to add tasks to the list, mark tasks as completed, and display the to-do list.
"""

class Task:
    def __init__(self, description, priority):
        self.description = description
        self.priority = priority
        self.next = None
        
class ToDoList:
    def __init__(self):
        self.head = None
        
    def add_task(self, description, priority):
        new_task = Task(description, priority)
        if self.head is None:
            self.head = new_task
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_task
            
    def complete_task(self, description):
        if self.head is None:
            print("ToDo list is empty")
            return
        if self.head.description == description:
            self.head = self.head.next
            return
        current = self.head
        while current.next:
            if current.next.description == description:
                current.next = current.next.next
                return
        print(f"Task '{description}' not found in the to-do list." )            
        
    def display_todo_list(self):
        
        if self.head is None:
            print("To-do list is empty.")
            return
        current = self.head
        while current:
            print(f"Description: {current.description}, Priority: {current.priority}")
            current = current.next        

# Creating a to-do list
todo_list = ToDoList()

# Adding tasks to the to-do list
todo_list.add_task("Complete assignment", "High")
todo_list.add_task("Grocery shopping", "Medium")
todo_list.add_task("Pay bills", "Low")

# Displaying the to-do list
print("To-Do List:")
todo_list.display_todo_list()

# Completing a task
todo_list.complete_task("Grocery shopping")

# Displaying the to-do list after completion
print("\nTo-Do List after completing 'Grocery shopping':")
todo_list.display_todo_list()



"""
Explanation
- We define a `Task` class to represent a task with attributes `description`, `priority`, and `next` pointer.
- We have a `ToDoList` class managing the to-do list using a singly linked list. It has methods to add a task, mark a task as completed, and display the to-do list.
- In the example, we create a to-do list object `todo_list` and add three tasks to it using the `add_task` method.
- We then display the to-do list using the `display_todo_list` method.
- Next, we mark a task titled "Grocery shopping" as completed using the `complete_task` method.
- Finally, we display the to-do list again to see the updated version after completion.
"""
 "Here's a simple C code snippet to demonstrate pointers:"

#include <stdio.h>

int main() {
    int num = 10;   // Declare an integer variable
    int *ptr;       // Declare a pointer variable

    ptr = &num;     // Assign the address of num to ptr

    printf("Value of num: %d\n", num);    // Print the value of num
    printf("Address of num: %p\n", &num);  // Print the address of num
    printf("Value of num via pointer: %d\n", *ptr);  // Print the value of num via pointer
    printf("Address stored in pointer: %p\n", ptr);  // Print the address stored in the pointer

    return 0;
}

"""
Explanation:
- `int num = 10;`: Declares an integer variable `num` and initializes it to `10`.
- `int *ptr;`: Declares an integer pointer variable `ptr`.
- `ptr = &num;`: Assigns the address of `num` to the pointer `ptr`.
- `*ptr`: Dereferences the pointer, giving the value stored at the memory address it points to.
- `printf()` statements: Print the value of `num`, address of `num`, value of `num` via pointer, and the address stored in the pointer `ptr`.

This code demonstrates how pointers work by showing how you can store the address of a variable in a pointer and then access the value of that variable indirectly through the pointer.
"""
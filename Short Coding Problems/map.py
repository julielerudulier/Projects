# 1st example: Applying a function to a single iterable
# Define a function that returns the square of an integer
def square(n):
    return n ** 2

# Create a list of numbers
lst = [1, 2, 3, 4, 5]

# Use map() to apply the square function to each element
squared_numbers = map(square, lst)

# Convert the result into a list
result = list(squared_numbers)
print(squared_numbers)

# The output is: [1, 4, 9, 16, 25]

# ---

# 2nd example: Multiple iterable mapping
# Define a function that adds two elements together
def add(x, y):
    return x + y

# Create two lists that we're going to add together
list1 = [1, 2, 3, 4, 5]
list2 = [10, 20, 30, 40, 50]

# Use map() to apply the add function to elements of both lists
result = list(map(add, list1, list2))
print(result)

# The output is: [11, 22, 33, 44, 55]

# ---

# 3rd example: Lambda functions and map()
# Following up on example #1, we can use a lambda function to square each element in a list and obtain the same result as with a function
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x ** 2, numbers))
print(squared_numbers)

# The output is: [1, 4, 9, 16, 25]

# ---

# 4th example: Combining map() with other functions
# In this example, we will use map() in conjunction with filter() to filter elements from a list based on a certain condition
# Define a function that returns the square of a given integer
def square(n):
    return n ** 2

# Define a second function that returns where an integer is odd or not
def is_odd(n):
    return n % 2 == 1

# Create a list of integers
numbers = [2, 1, 3, 4, 7, 11, 18]

# Create a list of squared odd numbers
result = list(map(square, filter(is_odd, numbers)))
print(result)

# Example 1: Applying a function to a single iterable
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

# Example 2: Multiple iterable mapping
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

# Example 3: Lambda functions and map()
# Following up on example #1, we can use a lambda function to square each element in a list and obtain the same result as with a function
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x ** 2, numbers))
print(squared_numbers)

# The output is: [1, 4, 9, 16, 25]

# ---

# Example 4: Combining map() with other functions
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

# ---

# Example 5: Here we want to extract the email adresses of customers from a CSV file
# We also want to remove any duplicates.
import csv
with open('customers.csv', 'r') as file:
    reader = csv.DictReader(file)
    emails = set(filter(lambda x: len(x) > 0, map(lambda row: row['email'], reader)))
print(emails)

# ---

# Example 6: In this example we have a text file containing a list of words 
# and we want to count the number of occurrences of each word
from functools import reduce

# Open file and create a list of ever word contained in the file
with open('words.txt', 'r') as file:
    words = file.read().split()

# Count the occurrences of each word using map() and reduce()
word_counts = reduce(lambda d, word: {**d, word: d.get(word, 0) + 1}, words, {})
print(word_counts)

# ---

# Example 7: Here we have a list of strings representing numbers
# and we want to convert them to integers and compute their sum
# Create list of numbers as strings
numbers = ['1', '2', '3', '4', '5']

# Convert the numbers to int type
integers = map(int, numbers)

# Compute the sum using reduce()
total = reduce(lambda x, y: x + y, integers)
print(total)

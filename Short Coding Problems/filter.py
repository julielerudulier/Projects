# Example 1: We have a list of numbers and we want to filter out the even ones:
# Create a list of numbers
numbers = [1, 2, 3, 4, 5]

# Create a new list with odd numbers only
odds = list(filter(lambda x: x % 2 != 0, numbers))
print(odds)

# The outcome is: [1, 3, 5]

# ---

# Example 2: Here we will remove empty strings from a list of strings:
# Create a list of words and empty strings
strings = ["hello", "", "world", "", "python"]

# Create a new list that only contains words 
non_empty = list(filter(lambda x: len(x) > 0, strings))
print(non_empty)

# The outcome is: ['hello', 'world', 'python']

# ---

# Example 3: In this example, we combine the map() function in conjunction with filter() 
# to filter elements from a list based on a certain condition:
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

# Example 4: Here we want to extract the email adresses of customers from a CSV file
# We also want to remove any duplicates.
import csv

# Open and save file and emails
with open('customers.csv', 'r') as file:
    reader = csv.DictReader(file)
    emails = set(filter(lambda x: len(x) > 0, map(lambda row: row['email'], reader)))
print(emails)

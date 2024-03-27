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

# Example 1: We have a list of numbers and we want to calculate their sum using reduce()
from functools import reduce

# Create list of numbers
numbers = [1, 2, 3, 4, 5]
total = reduce(lambda x, y: x + y, numbers)
print(total)

# The outcome is: 15

# ---

# Example 2: In a similar situation, suppose we have a list of strings representing numbers
# and we want to convert them to integers and compute their sum

# Create a list of numbers as strings
numbers = ['1', '2', '3', '4', '5']

# Convert the strings to integers using map()
integers = map(int, numbers)

# Compute the sum using reduce()
total = reduce(lambda x, y: x + y, integers)
print(total)

# The outcome is: 15

# ---

# Example 3: Here we want to find the maximum value in a list
# Create a list of numbers
numbers = [1, 7, 3, 9, 5]

# Find the biggest number of the list
max_number = reduce(lambda x, y: x if x > y else y, numbers)
print(max_number)

# The outcome is: 9

# ---

# Example 4: We have a list of numbers representing the grades of students in a class
# and we want to compute some statistics, such as the mean, median, and standard deviation
from functools import reduce
from statistics import median, stdev

grades = [75, 85, 90, 65, 80]

# Compute the mean using reduce()
total = reduce(lambda x, y: x + y, grades)
mean_grade = total / len(grades)
print("The mean grade is:", int(mean_grade))

# Compute the median using median()
median_grade = median(grades)
print("The median grade is:", median_grade)

# Compute the standard deviation using stdev()
std_dev = stdev(grades)
print("The standard deviation is:", round(std_dev, 1))

# ---

# Example 5: In this example we have a text file containing a list of words 
# and we want to count the number of occurrences of each word
# Open file and create a list of ever word contained in the file
with open('words.txt', 'r') as file:
    words = file.read().split()

# Count the occurrences of each word using map() and reduce()
word_counts = reduce(lambda d, word: {**d, word: d.get(word, 0) + 1}, words, {})
print(word_counts)

import numpy as np

def find_largest_square(arr):
    largest = arr[0]
    for i in arr:
        if i > largest:
            largest = i
    return largest **2

array = np.random.randint(0, 1000, 10)
largest_squared_number = find_largest_square(array)
print(f"The largest squared number in the list is {largest_squared_number}")

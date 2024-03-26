import numpy as np

def find_largest_squared(arr):
    largest = arr[0]
    for i in arr:
        if i > largest:
            largest = i
    return largest, largest **2

array = np.random.randint(0, 1000, 20)
largest, squared = find_largest_squared(array)
print(f"The largest number in the array is {largest}, and its square is {squared}.")

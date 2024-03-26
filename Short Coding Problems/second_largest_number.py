import numpy as np

def find_second_largest_num(lst):
    if len(lst) < 2:
        return "The list should contain at least two elements."

    unique_lst = list(set(lst)) 
    unique_lst.sort(reverse = True) 

    if len(unique_lst) >= 2:
        return unique_lst[1] 
    else:
        "There is no second-largest number in this list."

lst = np.random.randint(1, 100, 10)
second_largest_number = find_second_largest_num(lst)

print(lst)
print(f"The second-largest number is: {second_largest_number}.")

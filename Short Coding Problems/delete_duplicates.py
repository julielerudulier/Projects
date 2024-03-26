import numpy as np

def delete_duplicates(lst):
    new_lst = []
    for i in lst:
        if i not in new_lst:
            new_lst.append(i)
    return new_lst

lst = [1, 77, 38, 29, 77, 4, 62, 29, 56, 32, 1]
new_lst = delete_duplicates(lst)
reverse_lst = sorted(delete_duplicates(lst), reverse = True)

print(f"The initial list is: {lst}")
print(f"The list without duplicates is: {new_lst}")
print(f"The list without duplicates in reverse order is: {reverse_lst}")

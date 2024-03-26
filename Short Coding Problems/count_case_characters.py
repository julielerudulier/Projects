def count_case_characters(txt):
    upper = 0
    lower = 0

    for char in txt:
        if char.isupper():
            upper += 1
        elif char.islower():
            lower += 1
    return upper, lower

txt = input("Please write characters in both lower and upper case: ")
upper, lower = count_case_characters(txt)

print(f"The number of uppercase characters is: {upper}")
print(f"The number of lowercase characters is: {lower}")

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

x = int(input("Enter a number: "))
result = factorial(x)
print(f"The factorial of {x} is {result}")

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n **.5+1)): 
        if n % i == 0:
            return False
    return True
  
x = int(input("Enter a number: "))
if is_prime(x):
    print(f"The number {x} is a prime number.")
else:
    print(f"The number {x} is not a prime.")

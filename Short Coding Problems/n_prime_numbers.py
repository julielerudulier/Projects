def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n **0.5 - 1)):
        if n % i == 0:
            return False
    return True

def number_prime_number(n):
    count = 0
    for i in range(n):
        if is_prime(i):
            count += 1
    return count

n = int(input("Enter a number: "))
count = number_prime_number(n)

print(f"The number of prime numbers that are less than {n} is {count}.")

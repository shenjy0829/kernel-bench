def get_prime_factors(n):
    factors = set()

    # Handle the factor 2
    while n % 2 == 0:
        factors.add(2)
        n //= 2

    # Handle odd factors starting from 3
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors.add(i)
            n //= i
        i += 2
    print(list)
    # If n is still greater than 1, the remaining n must be prime
    if n > 1:
        factors.add(n)

    return sorted(list(factors))

# Example usage
number = 15
print(f"Prime factors of {number} are: {get_prime_factors(number)}")
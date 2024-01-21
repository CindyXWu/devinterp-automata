from lempel_ziv_complexity import lempel_ziv_complexity as lz
import time
import numpy as np

def lempel_ziv_complexity(bit_string):
    # Create an empty set to store substrings
    substrings = set()
    
    i = 0
    while i < len(bit_string):
        # Start with the first character
        for j in range(i+1, len(bit_string) + 1):
            # Extract the substring
            substring = bit_string[i:j]
            
            # If the substring is not in the set, add it
            if substring not in substrings:
                substrings.add(substring)
                break
        # Move the index
        i = j
        
    # The complexity is the number of substrings
    return len(substrings)

if __name__ == "__main__":
    bit_string = ''
    bits = np.random.randint(0,2,1000000)
    for bit in bits:
        bit_string += str(bit)
    start_time = time.time()
    lempel_ziv_complexity(bit_string)
    end_time = time.time()
    print(end_time-start_time)
    start_time = time.time()
    lz(bit_string)
    end_time = time.time()
    print(end_time-start_time)
    print(lz(bit_string), lempel_ziv_complexity(bit_string))
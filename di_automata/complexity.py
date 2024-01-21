from quine_mccluskey.qm import QuineMcCluskey
import math
from typing import List


def get_boolean_complexity(boolean_string: str) -> int:
    """For Majority function we get {'11-', '1-1', '-11'} which is correct, and corresponds to 5 gates:
    Each term (prime implicant) is an 'AND' expression between two bits. 
    (if there were 3 1s, i.e. '111-' then we would have 2 AND gates)
    Then, combine these terms with OR gates (2 in total here).

    To use this in future, just pass in a COMPLETE specification of all boolean input states that map to 1s in the list ones.
    
    Args:
        ones: a list of decimal ordered inputs (e.g. 3 means '11') that map to 1 (min terms).
        dc: a list of 'don't care' inputs (see QuineMcCluskey and Karnaugh maps).
    Returns:
        The minimum number of AND, OR and NOT gates required to implement the Boolean function.
    """
    if check_bs(boolean_string):
        # NOR - 1 NOT + (n-1) OR gates = n gates total
        return 2*math.log2(len(boolean_string))-2
    qm = QuineMcCluskey(use_xor=True)
    ones: List[int] = [i for i, bit in enumerate(boolean_string) if bit == '1']
    res = qm.simplify(ones, dc = [])
    # Remove dashes, concatenate results. There is a gate between every non-dash character now.
    concat_res = ''.join(s.replace('-', '') for s in res)
    n = len(concat_res) + concat_res.count('0')
    # print(f"Boolean string: {boolean_string}, Quine-McCluskey prime implicant form: {res}")
    return n-1 if n != 0 else 0


def check_bs(boolean_string: str) -> bool:
    """Checks if string has form leading 1 followed by all 0s."""
    return boolean_string.startswith('1') and all(c == '0' for c in boolean_string[1:])
    
    
def get_entropy(boolean_string: str) -> float:
    """Args:
        s: a list of Boolean string outputs on the test set (defines function).
    Returns:
        The binary entropy of a string, where p is given by the probability of a 1.
    """
    p, l = boolean_string.count('0') / len(boolean_string), boolean_string.count('1') / len(boolean_string)
    return -(p * math.log2(p) if p > 0 else 0) - (l * math.log2(l) if l > 0 else 0)
    

def get_lempel_ziv(boolean_string: str) -> float:

    length = len(boolean_string)
    if '1' not in boolean_string or '0' not in boolean_string:
        return math.log2(length)
    
    n_substrings = 0
    for string in (boolean_string, boolean_string[::-1]):
        i = 0
        substrings = set()
        while i < length:
            # Start with irst character
            for j in range(i+1, len(string) + 1):
                substring = string[i:j]
                if substring not in substrings:
                    substrings.add(substring)
                    break
            # Move index
            i = j
            n_substrings += len(substrings)/2
        
    # Complexity is number of substrings
    return math.log2(length) * n_substrings


def get_csr(boolean_string: str) -> float:
    pass


def test_qm():
    qm = QuineMcCluskey(use_xor=True)
    # Test on Majority function
    ones = ['011', '101', '110', '111']
    ones = [0]
    dc = []
    res = qm.simplify(ones, dc)
    print(res)

    
if __name__ == '__main__':
    # print(get_boolean_complexity('10'))
    test_qm()

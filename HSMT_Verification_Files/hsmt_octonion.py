import numpy as np
from typing import List, Tuple

class Octonion:
    """
    Octonion class with full Fano-plane multiplication table.
    Basis: 1, e1, e2, e3, e4, e5, e6, e7
    """
    def __init__(self, coeffs: np.ndarray):
        self.c = np.asarray(coeffs, dtype=complex)
        assert len(self.c) == 8, "Octonion must have 8 coefficients"

    def __add__(self, other):
        return Octonion(self.c + other.c)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return Octonion(self.c * other)
        a = self.c
        b = other.c
        c = np.zeros(8, dtype=complex)
        
        # Real part
        c[0] = a[0]*b[0] - np.dot(a[1:], b[1:])
        
        # Imaginary parts - Standard Fano plane multiplication
        # e1*e2=e3, e1*e4=e5, etc. with signs
        fano_table = [
            #   e1   e2   e3   e4   e5   e6   e7
            [ 0,  3, -2,  5, -4, -7,  6],  # *e1
            [ -3, 0,  1,  6, -7,  4, -5],  # *e2
            [  2, -1, 0,  7,  6, -5, -4],  # *e3
            [ -5, -6, -7, 0,  1,  2,  3],  # *e4
            [  4,  7, -6, -1, 0, -3,  2],  # *e5
            [  7, -4,  5, -2,  3, 0, -1],  # *e6
            [ -6,  5,  4, -3, -2,  1, 0]   # *e7
        ]
        
        for i in range(1,8):
            for j in range(1,8):
                k = abs(fano_table[i-1][j-1])
                sign = np.sign(fano_table[i-1][j-1])
                c[k] += sign * a[i] * b[j]
        
        return Octonion(c)

    def __rmul__(self, other):
        return self.__mul__(other)

    def conj(self):
        return Octonion(np.array([self.c[0]] + [-x for x in self.c[1:]]))

    def norm(self):
        return np.sum(np.abs(self.c)**2)

    def __repr__(self):
        return f"Octonion({self.c})"

# Left and Right multiplication operators
def left_mult(u: Octonion, v: Octonion) -> Octonion:
    return u * v

def right_mult(u: Octonion, v: Octonion) -> Octonion:
    return v * u

print("Octonion class defined successfully.")
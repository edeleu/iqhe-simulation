# normalized distribution for later fitting

import numpy as np
from scipy.special import gamma

def norm_coeffs(n,y):
    """Returns A,B such that the distribution has unit mean and is a proper PDF."""
    gamma1 = gamma((1+n) / y)
    gamma2 = gamma((2+n) / y)

    B = (gamma2/gamma1)**(y)
    A = (y*B**((1+n)/y)) / gamma1

    return A, B

def normalized_pdf(s, n, y):
    """
    PDF of the normalized distribution:
    f(s; n, y) = A(n,y) * s^n * exp(-B(n,y) * s^y)
    where A and B ensure normalization and unit mean.

    s, y >= 0, n >= -1
    """
    
    A, B = norm_coeffs(n, y)
    return A * s**n * np.exp(-B * s**y)

# Test values against known GUE results 
n=2 # Beta
y=2 #Exp Decay
A, B = norm_coeffs(n, y)

print(-B)
print("expected:", -4/np.pi)

print(A)
print("expected,",32/(np.pi*np.pi))
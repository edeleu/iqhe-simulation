# normalized distribution for later fitting

import numpy as np
from scipy.special import gamma

def c_of_b(b):
    """Returns c(b) such that the distribution has unit mean."""
    gamma1 = gamma((b + 2) / 2)
    gamma2 = gamma((b + 1) / 2)
    return (gamma1 / gamma2)**2

def normalized_pdf(s, b):
    """
    PDF of the normalized distribution:
    f(s; b) = A(b) * s^b * exp(-c(b) * s^2)
    where A(b) ensures normalization and unit mean.
    """
    if b <= -1 or np.any(s <= 0):
        return np.zeros_like(s)
    
    c = c_of_b(b)
    A = 2 * c**((b + 1)/2) / gamma((b + 1)/2)
    return A * s**b * np.exp(-c * s**2)

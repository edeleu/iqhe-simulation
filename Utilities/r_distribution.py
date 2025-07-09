import numpy as np

def generalized_r(r, b=2):
    """
    PDF of the general r-distribution
    where b is the Wigner beta value.
    """
    normconstant = (4/81) * np.pi / np.sqrt(3)
    numerator = (r + r**2)**b
    denominator = (1 + r + r**2)**(1 + 3*b/2)
    return numerator / (normconstant * denominator)

def generalized_folded_r(r, b=2):
    """
    folded, normalized PDF of the general r-distribution
    r here is r_tilda (min-max) from 0 --> 1
    """

    return generalized_r(r,b) + (1/(r**2))*generalized_r(1/r,b)

import matplotlib.pyplot as plt

r_vals = np.linspace(0.01, 1, 500)
plt.plot(r_vals, generalized_folded_r(r_vals), label='Folded GUE (β=2)')
plt.xlabel(r'$\tilde{r}$')
plt.ylabel(r'$\tilde{P}(\tilde{r})$')
plt.title("Folded Wigner-Dyson Distribution")
plt.legend()
plt.grid(True)
plt.show()

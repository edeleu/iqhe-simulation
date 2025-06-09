import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpmath import mp, mpc, exp, pi

# Set precision
mp.dps = 25

# Physical parameters
N_phi = 16
Lx = Ly = np.sqrt(2 * np.pi * N_phi)
tau = 1j * Ly / Lx
tau_mp = mpc(0, Ly / Lx)  # mpmath complex form

# Sample normalized eigenvector (replace with your own)
rng = np.random.default_rng(0)
c = rng.normal(size=N_phi) + 1j * rng.normal(size=N_phi)
c /= np.linalg.norm(c)

# Grid
Nx, Ny = 60, 60
x_vals = np.linspace(0, Lx, Nx)
y_vals = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
Z = X + 1j * Y

# --- Jacobi theta function with characteristics ---
def theta_char(z, tau, a=0, b=0, N_terms=10):
    result = mp.mpc(0)
    for n in range(-N_terms, N_terms + 1):
        n_shift = n + a
        exponent = pi * 1j * n_shift**2 * tau + 2 * pi * 1j * n_shift * (z + b)
        result += exp(exponent)
    return result

# --- Compute wavefunction ψ(x,y) on the torus ---
psi = np.zeros_like(Z, dtype=complex)

for m in range(N_phi):
    a = mp.mpf(m) / N_phi
    for i in range(Nx):
        for j in range(Ny):
            x = x_vals[i]
            y = y_vals[j]
            z = mpc(x, y)
            z_scaled = N_phi * z / Ly

            # Magnetic translation phase factor
            mag_phase = exp(pi * 1j * N_phi * x * y / (Lx * Ly))

            # Theta function with characteristic
            theta_val = theta_char(z_scaled, N_phi * tau_mp, a=a, b=0)

            # Add basis contribution
            psi[i, j] += c[m] * complex(mag_phase * theta_val)

# Normalize ψ
psi /= np.sqrt(np.trapz(np.trapz(np.abs(psi)**2, x_vals), y_vals))

# Plot |ψ(x, y)|²
plt.figure(figsize=(6, 5))
plt.pcolormesh(x_vals, y_vals, np.abs(psi.T)**2, shading='auto', cmap='inferno',
               norm=colors.Normalize(vmin=0, vmax=np.max(np.abs(psi)**2)))
plt.colorbar(label=r'$|\psi(x, y)|^2$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('LLL Torus Wavefunction using Jacobi Theta with Characteristics')
plt.tight_layout()
plt.show()

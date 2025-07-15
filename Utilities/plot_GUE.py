import numpy as np
import matplotlib.pyplot as plt

# def overlay_gue_curve(ax, s_max=6, num_points=1000, label="Reference GUE", color="green", linestyle="--"):
s = np.linspace(1e-5, 5, 1000)
p_s = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
    # p_s = np.exp(-1.65*(s-0.6))
    # p_s = s*np.exp(-s)
plt.plot(s, p_s)
plt.yscale('log')
plt.xscale('log')
plt.show()

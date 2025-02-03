# Integer Quantum Hall Effect Simulation

A numerical study of the lowest Landau level, simulating a 2D system under high magnetic field with a Gaussian White noise random potential. An optimized Python-based approach for solving eigenstates of the system's Hamiltonian for the analysis of Chern number statistics.

Guide of Files:
- Simulation.py: Main entry point to running simulations or ensembles and saving chern numbers.
- Simulation_GPU.py: The same simulation code, with some modifications to allow for speed-ups using GPU for linear algebra
- Helpers.py: Some helper functions for plotting, modifying eigenvectors, etc.
- precompute.py: A helper file to precompute common exponentials when using a fixed theta-resolution for improved efficiency
- AutoConverve.py: A file utilizing binary-search to find the theta-resolution where chern numbers converge
- viewMismatch.py: A file for viewing surfaces of eigenvalues where mismatching chern-numbers occur (due to insufficient theta-resolution).

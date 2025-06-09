# Integer Quantum Hall Effect Simulation

A numerical study of the lowest Landau level, simulating a 2D system under high magnetic field with a Gaussian White noise random potential. An optimized Python-based approach for solving eigenstates of the system's Hamiltonian for the analysis of Chern number statistics.

Guide of Files:
- Simulation.py: Main entry point to running simulations or ensembles and saving chern numbers.
- SimulationFULLGPU.py: The same simulation code, with some modifications to allow for speed-ups using GPU for linear algebra.
- SimulationMemEfficientGPU.py: The same simulation code, with some modifications to allow for speed-ups using GPU for linear algebra. This limits the amount of CPU memory used while offloading data to GPU memory (which may be limited as appropriate)
- Helpers.py: Some helper functions for plotting, modifying eigenvectors, etc.
- precompute.py: A helper file to precompute common exponentials when using a fixed theta-resolution for improved efficiency.

Analysis: Contains scripts for analyzing, plotting, and visualizing the data in various automated ways.

Convergence: Contains scripts for analyzing the boundary-space resolution required for converged Chern numbers.

Experimental: Various experimental scripts not necessarily in working-order.
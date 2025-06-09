import numpy as np
import pandas as pd
import csv, os
from numba import njit, prange, jit
import matplotlib.pyplot as plt
import random

# Define system-wide parameters...
# num-states is defined according to the ratio  
# N= B*L^2 / phi_o for phi_o=hc/e OR
# N = A / (2pi ell^2)=> so L^2 = 2pi*ell^2*N
# for ell^2 = hbar c / (e*B)

IMAG = 1j
PI = np.pi

NUM_STATES = 64     # Number of states for the system
NUM_THETA=26        # Number of theta for the THETA x,y mesh
POTENTIAL_MATRIX_SIZE = int(4*np.sqrt(NUM_STATES))
# basically experimental parameter, complete more testing with this size... (keep L/M small)

# Construct the Hamiltonian Matrix for a given Theta, Num_States, and random-potential.
# note: "theta" is actually an index here for efficiency
@njit(parallel=True,fastmath=True)
def constructHamiltonian(matrix_size, theta_x, theta_y, matrixV):
    # define values as complex128, ie, double precision for real and imaginary parts
    H = np.zeros((matrix_size,matrix_size),dtype=np.complex128)

    # actually, we only require the lower-triangular portion of the matrix, since it's Hermitian, taking advantage of eigh functon!
    for j in prange(matrix_size):
        for k in range(j+1):
            # j,k entry in H
            H[j,k]=constructHamiltonianEntryOriginal(j,k,theta_x,theta_y,matrixV)
    return H

# a simplified "original" version of constructing the entries, a bit easier to understand what's going on than above
@njit()
def constructHamiltonianEntryOriginal(indexJ,indexK,theta_x,theta_y,matrixV):
    value = 0
    
    for n in range(-POTENTIAL_MATRIX_SIZE, POTENTIAL_MATRIX_SIZE+1):
        if deltaPBC(indexJ,indexK-n):
            for m in range(-POTENTIAL_MATRIX_SIZE, POTENTIAL_MATRIX_SIZE+1):
                potentialValue = matrixV[m+POTENTIAL_MATRIX_SIZE,n+POTENTIAL_MATRIX_SIZE]
                exp_term = (1/NUM_STATES)*((-PI/2)*(m**2+n**2)-IMAG*PI*m*n+IMAG*(m*theta_y-n*theta_x)-IMAG*m*2*PI*indexJ)
                value +=potentialValue*np.exp(exp_term)
    return value

# periodic boundary condition for delta-fx inner product <phi_j | phi_{k-n}> since phi_j=phi_{j+N}
@njit()
def deltaPBC(a,b,N=NUM_STATES):
    modulus = (a-b)%N
    if modulus==0: return 1
    return 0

# -----------------------------------------------------------------------------
# 1. Compute the Berry phase on a single cell (for all bands)
# -----------------------------------------------------------------------------
def compute_cell_berry_phase(V, theta_x, theta_y, dtheta):
    """
    Compute the Berry phase (for each band) for a cell (plaquette) defined by the 
    lower-left corner (theta_x, theta_y) and side length dtheta.
    
    Returns:
      berry_phase : a numpy array of shape (NUM_STATES,) with the Berry phase for each band.
    """
    # Evaluate the Hamiltonian and eigenvectors at the four corners:
    # Lower-left corner:
    H_ll = constructHamiltonian(NUM_STATES, theta_x, theta_y, V)
    eigs_ll, eigv_ll = np.linalg.eigh(H_ll)
    
    # Lower-right corner:
    H_lr = constructHamiltonian(NUM_STATES, theta_x + dtheta, theta_y, V)
    eigs_lr, eigv_lr = np.linalg.eigh(H_lr)
    
    # Upper-right corner:
    H_ur = constructHamiltonian(NUM_STATES, theta_x + dtheta, theta_y + dtheta, V)
    eigs_ur, eigv_ur = np.linalg.eigh(H_ur)
    
    # Upper-left corner:
    H_ul = constructHamiltonian(NUM_STATES, theta_x, theta_y + dtheta, V)
    eigs_ul, eigv_ul = np.linalg.eigh(H_ul)
    
    # Compute inner products for each band.
    # The operations below produce an array of length NUM_STATES.
    ip1 = np.sum(np.conj(eigv_ll) * eigv_lr, axis=0)
    ip2 = np.sum(np.conj(eigv_lr) * eigv_ur, axis=0)
    ip3 = np.sum(np.conj(eigv_ur) * eigv_ul, axis=0)
    ip4 = np.sum(np.conj(eigv_ul) * eigv_ll, axis=0)
    
    # Berry phase for each band: take the angle of the product of inner products.
    berry_phase = np.angle(ip1 * ip2 * ip3 * ip4)
    return berry_phase

# -----------------------------------------------------------------------------
# 2. Adaptive refinement on a single cell (for all bands)
# # -----------------------------------------------------------------------------
# def adaptive_cell(V, theta_x, theta_y, dtheta, tol, max_depth, depth=0):
#     """
#     Recursively computes the Berry phase for a cell using adaptive mesh refinement.
    
#     Parameters:
#       V         : Potential matrix.
#       theta_x   : Lower-left x-coordinate of the cell.
#       theta_y   : Lower-left y-coordinate of the cell.
#       dtheta    : Side length of the cell.
#       tol       : Tolerance for the difference between the coarse and refined estimates.
#       max_depth : Maximum recursion depth.
#       depth     : Current recursion depth.
      
#     Returns:
#       phase     : A numpy array of shape (NUM_STATES,) containing the refined Berry phase for each band.
#     """
#     # Coarse estimate: compute the Berry phase over the entire cell.
#     p_coarse = compute_cell_berry_phase(V, theta_x, theta_y, dtheta)
    
#     if depth >= max_depth:
#         return p_coarse
    
#     # Subdivide the cell into 4 equal subcells.
#     dtheta_half = dtheta / 2.0
#     p1 = adaptive_cell(V, theta_x,               theta_y,               dtheta_half, tol, max_depth, depth+1)
#     p2 = adaptive_cell(V, theta_x + dtheta_half, theta_y,               dtheta_half, tol, max_depth, depth+1)
#     p3 = adaptive_cell(V, theta_x + dtheta_half, theta_y + dtheta_half, dtheta_half, tol, max_depth, depth+1)
#     p4 = adaptive_cell(V, theta_x,               theta_y + dtheta_half, dtheta_half, tol, max_depth, depth+1)
    
#     # Refined phase is the sum over the four subcells (vector addition).
#     p_refined = p1 + p2 + p3 + p4
    
#     # Compute the error as the maximum absolute difference over all bands.
#     error = np.max(np.abs(p_refined - p_coarse))
#     if error < tol:
#         return p_refined
#     else:
#         print("ERROR:", error, "at thetax", theta_x, "thetaY", theta_y)

#         return p_refined  # Alternatively, one could return p_coarse if max_depth not reached.
    

def adaptive_cell(V, theta_x, theta_y, dtheta, tol, max_depth, depth=0):
    """
    Recursively compute the Berry phase on a cell using adaptive refinement, only refining
    when the error exceeds the given tolerance.
    
    Parameters:
      V         : Potential matrix.
      theta_x   : Lower-left x-coordinate of the cell.
      theta_y   : Lower-left y-coordinate of the cell.
      dtheta    : Side length of the cell.
      tol       : Tolerance for the difference between coarse and refined estimates.
      max_depth : Maximum recursion depth.
      depth     : Current recursion depth.
      
    Returns:
      A numpy array of shape (NUM_STATES,) with the Berry phase contribution from the cell.
    """
    # Compute the coarse estimate over the entire cell
    p_coarse = compute_cell_berry_phase(V, theta_x, theta_y, dtheta)

    # Base case: if we hit max depth, return the coarse estimate
    if depth >= max_depth:
        return p_coarse

    # Compute refined estimate by subdividing the cell
    dtheta_half = dtheta / 2.0
    p1 = compute_cell_berry_phase(V, theta_x,               theta_y,               dtheta_half)
    p2 = compute_cell_berry_phase(V, theta_x + dtheta_half, theta_y,               dtheta_half)
    p3 = compute_cell_berry_phase(V, theta_x + dtheta_half, theta_y + dtheta_half, dtheta_half)
    p4 = compute_cell_berry_phase(V, theta_x,               theta_y + dtheta_half, dtheta_half)

    # Sum the contributions from the subcells
    p_refined = p1 + p2 + p3 + p4

    # Compute the error between refined and coarse values
    error = np.max(np.abs(phase_diff(p_refined, p_coarse)))  # Ensure phase continuity

    # If error is within tolerance, accept the coarser estimate
    if error < tol:
        return p_coarse
    
    # Otherwise, refine further
    p1 = adaptive_cell(V, theta_x,               theta_y,               dtheta_half, tol, max_depth, depth + 1)
    p2 = adaptive_cell(V, theta_x + dtheta_half, theta_y,               dtheta_half, tol, max_depth, depth + 1)
    p3 = adaptive_cell(V, theta_x + dtheta_half, theta_y + dtheta_half, dtheta_half, tol, max_depth, depth + 1)
    p4 = adaptive_cell(V, theta_x,               theta_y + dtheta_half, dtheta_half, tol, max_depth, depth + 1)

    # Sum the refined contributions
    return p1 + p2 + p3 + p4

# -----------------------------------------------------------------------------
# 3. Integration over the full (0,2π)x(0,2π) domain using adaptive refinement.
# -----------------------------------------------------------------------------
def adaptiveChernNumber(V, tol=1e-3, max_depth=6, initial_grid=10):
    """
    Computes the Chern number via adaptive integration of the Berry phase over the full 
    parameter space [0, 2π] x [0, 2π] using adaptive mesh refinement (AMR) computed for all bands.
    
    Parameters:
      V           : Potential matrix.
      tol         : Tolerance for adaptive refinement.
      max_depth   : Maximum recursion depth.
      initial_grid: The initial number of cells per dimension.
    
    Returns:
      chern_numbers : A numpy array of shape (NUM_STATES,) containing the computed Chern numbers for each band.
    """
    dtheta0 = 2 * np.pi / initial_grid
    total_phase = np.zeros(NUM_STATES)  # For all bands
    
    # Loop over the initial grid of cells.
    for i in range(initial_grid):
        for j in range(initial_grid):
            theta_x = i * dtheta0
            theta_y = j * dtheta0
            cell_phase = adaptive_cell(V, theta_x, theta_y, dtheta0, tol, max_depth)
            total_phase += cell_phase
    
    # The Chern number for each band is the total Berry phase divided by 2π.
    chern_numbers = total_phase / (2 * np.pi)
    return chern_numbers + 1/NUM_STATES

def phase_diff(a, b):
    """
    Compute the difference a - b in a way that accounts for the 2pi periodicity.
    Returns a value in the interval [-pi, pi].
    """
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi
# -----------------------------------------------------------------------------
# 4. Example usage.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Construct the potential matrix V using your function.
    V = np.load("potential_matrix_trial_118_Nstates_64.npy")
    
    # Set parameters for adaptive integration.
    tol = 1e-2         # Tolerance for refinement.
    max_depth = 5      # Maximum recursion depth.
    initial_grid = 5  # Initial grid resolution (10x10 cells).
    
    chern = adaptiveChernNumber(V, tol=tol, max_depth=max_depth, initial_grid=initial_grid)
    print("Computed Chern numbers (adaptive refinement, all bands):", chern)

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

def fullSimulationGrid(V, thetaResolution=10,visualize=False):

    # next, loop over theta_x theta_y (actually just the indices)
    thetas = np.linspace(0, 2*PI, num=thetaResolution, endpoint=True)
    delTheta = thetas[1]
    # print(delTheta)

    eigGrid = np.zeros((thetaResolution,thetaResolution),dtype=object)
    # eigValueGrid = np.zeros((thetaResolution,thetaResolution, NUM_STATES,NUM_STATES),dtype=complex)
    for indexONE in range(thetaResolution):
        for indexTWO in range(thetaResolution):
            # H = constructHamiltonian(NUM_STATES,thetas[indexONE],thetas[indexTWO], V) 
            H = constructHamiltonian(NUM_STATES, thetas[indexONE], thetas[indexTWO], V)

            eigs, eigv = np.linalg.eigh(H,UPLO="L")
            # eigs, eigv = scipy.linalg.eigh(H,driver="evd")

            # eigGrid[indexONE,indexTWO]=[thetas[indexONE],thetas[indexTWO],eigs]
            eigGrid[indexONE,indexTWO]=[thetas[indexONE],thetas[indexTWO],eigs]   
            # eigValueGrid[indexONE,indexTWO]=eigv

    return eigGrid

    # cherns = np.round(computeChernGridV2_vectorized(eigValueGrid,thetaResolution),decimals=3)

    # if visualize:
    #     print(cherns)
    #     print("Sum",sum(cherns))
    #     helpers.plotEigenvalueMeshHelper(eigGrid,thetaResolution,NUM_STATES)

    # return cherns

def plotEigenvalueMeshHelper(grid, numTheta, N, mismatch_indices=None):
    """
    Plots the 8 eigenvalue surfaces nearest to mismatches, highlighting mismatches in RED.
    
    - grid: Eigenvalue data grid [theta_x, theta_y, eigs] at each (i, j)
    - numTheta: Number of theta points along one axis
    - N: Total number of eigenvalues
    - mismatch_indices: List of mismatched eigenvalue indices to highlight (default: center region)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate random colors for non-mismatched eigenvalues
    colors = [tuple(random.random() for _ in range(3)) for _ in range(N)]  

    # Locate eigenvalues closest to the mismatches
    X = np.array([[grid[i, j][0] for j in range(numTheta)] for i in range(numTheta)])
    Y = np.array([[grid[i, j][1] for j in range(numTheta)] for i in range(numTheta)])
    
    # **Step 1: Identify indices to plot**
    if mismatch_indices is None or len(mismatch_indices) == 0:
        mismatch_indices = [N // 2]  # Default to center eigenvalue index

    # Find the range of eigenvalues to plot (center around mismatches)
    min_idx = max(0, min(mismatch_indices) - 4)
    max_idx = min(N, max(mismatch_indices) + 4)

    for idx in range(min_idx, max_idx):  
        Z = np.array([[grid[i, j][2][idx] for j in range(numTheta)] for i in range(numTheta)])
        
        # Highlight mismatched eigenvalues in RED
        if idx in mismatch_indices:
            ax.plot_surface(X, Y, Z, color='red', edgecolor='none', alpha=0.8)
        else:
            ax.plot_surface(X, Y, Z, color=colors[idx], edgecolor='none', alpha=0.8)

    # Set axis labels
    ax.set_xlabel('Theta-X')
    ax.set_ylabel('Theta-Y')
    ax.set_zlabel('Energy Value')
    ax.set_title(f'Eigenvalue Surfaces Near Mismatch')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mismatch = []
    V_loaded = np.load("potential_matrix_trial_1_res_32.npy")
    grid = fullSimulationGrid(V_loaded,NUM_THETA)
    plotEigenvalueMeshHelper(grid,NUM_THETA,NUM_STATES,mismatch)
import numpy as np
import cupy as cp
import csv
import timeit
from numba import njit, prange, jit
from timeit import default_timer as timer

import precompute
import helpers

# Define system-wide parameters...
# num-states is defined according to the ratio  
# N= B*L^2 / phi_o for phi_o=hc/e OR
# N = A / (2pi ell^2)=> so L^2 = 2pi*ell^2*N
# for ell^2 = hbar c / (e*B)

IMAG = 1j
PI = np.pi

NUM_STATES = 128     # Number of states for the system
NUM_THETA=30        # Number of theta for the THETA x,y mesh
POTENTIAL_MATRIX_SIZE = int(4*np.sqrt(NUM_STATES))
# basically experimental parameter, complete more testing with this size... (keep L/M small)

# Precompute look-up tables (matrices) for Hamiltonian exponentials
mn_LUT, mj_LUT, mTheta_LUT = precompute.precompute_exponentialsNumPy(NUM_STATES, POTENTIAL_MATRIX_SIZE, NUM_THETA)

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
            H[j,k]=constructHamiltonianEntry(j,k,theta_x,theta_y,matrixV)
    return H

# Compute the value for a single entry of the Hamiltonian matrix, which is a summation... 
# @njit(boundscheck=True)
@njit()
def constructHamiltonianEntry(indexJ,indexK,theta_x,theta_y,matrixV):
    value = 0

    for n in range(-POTENTIAL_MATRIX_SIZE, POTENTIAL_MATRIX_SIZE+1):
        if deltaPBC(indexJ,indexK-n):
            for m in range(-POTENTIAL_MATRIX_SIZE, POTENTIAL_MATRIX_SIZE+1):
                value += matrixV[m+POTENTIAL_MATRIX_SIZE,n+POTENTIAL_MATRIX_SIZE]*mn_LUT[m+POTENTIAL_MATRIX_SIZE,n+POTENTIAL_MATRIX_SIZE]*mj_LUT[m+POTENTIAL_MATRIX_SIZE,indexJ]*mTheta_LUT[m+POTENTIAL_MATRIX_SIZE,theta_y]/mTheta_LUT[n+POTENTIAL_MATRIX_SIZE,theta_x]
    return value

# a simplified "original" version of constructing the entries, a bit easier to understand what's going on than above
# furthermore, we pass explicit thetas instead of indices
def constructHamiltonianOriginal(matrix_size, theta_x, theta_y, matrixV):
    # define values as complex128, ie, double precision for real and imaginary parts
    H = np.zeros((matrix_size,matrix_size),dtype=np.complex128)

    # actually, we only require the lower-triangular portion of the matrix, since it's Hermitian, taking advantage of eigh functon!
    for j in prange(matrix_size):
        for k in range(j+1):
            # j,k entry in H
            H[j,k]=constructHamiltonianEntryOriginal(j,k,theta_x,theta_y,matrixV)
    return H

def constructHamiltonianEntryOriginal(indexJ,indexK,theta_x,theta_y,matrixV):
    value = 0
    
    for n in range(-POTENTIAL_MATRIX_SIZE, POTENTIAL_MATRIX_SIZE+1):
        if deltaPBC(indexJ,indexK-n):
            for m in range(-POTENTIAL_MATRIX_SIZE, POTENTIAL_MATRIX_SIZE+1):
                potentialValue = matrixV[m+POTENTIAL_MATRIX_SIZE,n+POTENTIAL_MATRIX_SIZE]
                exp_term = (1/NUM_STATES)*((-PI/2)*(m**2+n**2)-IMAG*PI*m*n+IMAG*(m*theta_y-n*theta_x)-IMAG*m*2*PI*indexJ)
                value +=potentialValue*np.exp(exp_term)

                # alternatively break exponential into pieces, (slower, for testing)
                # exp_termONE = np.exp(-PI*(m**2+n**2)/(2*NUM_STATES))
                # exp_termTWO = np.exp(-IMAG*PI*m*n/NUM_STATES)
                # exp_termTHREE = np.exp(IMAG*(m*theta_y-n*theta_x)/NUM_STATES)
                # exp_termFOUR = np.exp(-IMAG*2*PI*m*indexJ/NUM_STATES)
                # value += potentialValue*exp_termONE*exp_termTWO*exp_termTHREE*exp_termFOUR
    return value

# periodic boundary condition for delta-fx inner product <phi_j | phi_{k-n}> since phi_j=phi_{j+N}
@njit()
def deltaPBC(a,b,N=NUM_STATES):
    modulus = (a-b)%N
    if modulus==0: return 1
    return 0

# Construct the potential's fourier coefficients as in Yan Huo's thesis... 
def constructPotential(size, mean=0, stdev=1):
    # define a real, periodic gaussian potential over a size x size field where
    # V(x,y)=sum V_{m,n} exp(2pi*i*m*x/L_x) exp(2pi*i*n*y/L_y)

    # we allow V_(0,0) to be at V[size,size] such that we can have coefficients from V_(-size,-size) to V_(size,size). We want to have both negatively and
    # positively indexed coefficients! 

    V = np.zeros((2*size+1, 2*size+1), dtype=complex)
    # loop over values in the positive +i,+j quadrant and +i,-j quadrant, assigning conjugates at opposite quadrants
    for i in range(size + 1):
        for j in range(-size, size + 1):
            # Real and imaginary parts
            real_part = np.random.normal(0, stdev)
            imag_part = np.random.normal(0, stdev)
            
            # Assign the complex value 
            V[size + i, size + j] = real_part + IMAG * imag_part
            
            # Enforce the symmetry condition, satisfy that V_{i,j}^* = V_{-i,-j}
            if not (i == 0 and j == 0):  # Avoid double-setting the origin
                V[size - i, size - j] = real_part - IMAG * imag_part

    # set origin equal to a REAL number! (DC OFFSET)
    # V[size, size] = np.random.normal(0, stdev) + 0*IMAG 

    # Set DC offset to zero so avg over real-space potential = 0
    V[size, size] = 0

    return V

# run a full iteration of the simulation. that is, compute the potential, then
# construct the hamiltonian and find the eigenvalues for a mesh of theta_x, theta_y's.
def fullSimulationGrid(thetaResolution=10,visualize=False):
    # for a given iteration, define a random potential...
    V = constructPotential(POTENTIAL_MATRIX_SIZE)

    # next, loop over theta_x theta_y (actually just the indices)

    # Pre-allocate space for Hamiltonians on the GPU
    H_gpu = cp.zeros((NUM_STATES, NUM_STATES), dtype=cp.complex128)

    eigValueGrid = np.zeros((thetaResolution,thetaResolution, NUM_STATES,NUM_STATES),dtype=np.complex128)
    for indexONE in range(thetaResolution):
        for indexTWO in range(thetaResolution):
            # H = constructHamiltonian(NUM_STATES,thetas[indexONE],thetas[indexTWO], V) 
            H = constructHamiltonian(NUM_STATES,indexONE,indexTWO, V) 
            H_gpu = cp.asarray(H)

            eigs_gpu, eigv_gpu = cp.linalg.eigh(H_gpu, UPLO="L")

            eigValueGrid[indexONE,indexTWO]=cp.asnumpy(eigv_gpu)

    # now, vectorized computation of chern-numbers!

    cherns = np.round(computeChernGridV2_vectorized(eigValueGrid,thetaResolution),decimals=3)

    if visualize:
        print(cherns)
        print("Sum",sum(cherns))

    return cherns

def fullSimulationGrid_batches(thetaResolution=10, batch_size=16, visualize=False):
    """
    Perform a simulation on a grid of (theta_x, theta_y) values using batched GPU eigendecomposition.
    
    Parameters:
        thetaResolution (int): Number of (theta_x, theta_y) grid points in each dimension.
        batch_size (int): Number of Hamiltonians to process in a single batch.
        visualize (bool): Whether to visualize the results (e.g., print Chern numbers).

    Returns:
        np.ndarray: Computed Chern numbers for the system.
    """
    # Generate the random potential
    V = constructPotential(POTENTIAL_MATRIX_SIZE)

    # Preallocate arrays for results
    eigValueGrid = np.zeros((thetaResolution, thetaResolution, NUM_STATES, NUM_STATES), dtype=np.complex128)

    # Precompute indices for batched processing
    theta_indices = [(i, j) for i in range(thetaResolution) for j in range(thetaResolution)]
    num_batches = int(np.ceil(len(theta_indices) / batch_size))

    for batch_idx in range(num_batches):
        # Get the current batch of (theta_x, theta_y) indices
        batch_indices = theta_indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_size_actual = len(batch_indices)

        # Preallocate batch of Hamiltonians on the CPU
        hamiltonians_cpu = np.zeros((batch_size_actual, NUM_STATES, NUM_STATES), dtype=np.complex128)

        # Construct Hamiltonians in parallel on the CPU
        for batch_local_idx, (indexONE, indexTWO) in enumerate(batch_indices):
            hamiltonians_cpu[batch_local_idx] = constructHamiltonian(NUM_STATES, indexONE, indexTWO, V)

        # Transfer batch of Hamiltonians to GPU
        hamiltonians_gpu = cp.asarray(hamiltonians_cpu)

        # Perform batched eigendecomposition on the GPU
        eigvals_gpu, eigvecs_gpu = cp.linalg.eigh(hamiltonians_gpu, UPLO="L")

        # Transfer eigenvectors back to CPU
        eigvecs_cpu = cp.asnumpy(eigvecs_gpu)

        # Store eigenvectors in eigValueGrid
        for batch_local_idx, (indexONE, indexTWO) in enumerate(batch_indices):
            eigValueGrid[indexONE, indexTWO] = eigvecs_cpu[batch_local_idx]

    # Compute Chern numbers (vectorized)
    cherns = np.round(computeChernGridV2_vectorized(eigValueGrid, thetaResolution), decimals=3)

    if visualize:
        print("Chern Numbers:", cherns)
        print("Sum of Chern Numbers:", sum(cherns))

    return cherns

#NJIT speedup only occurs when using ensemble, otherwise it actually slows down on the first pass...
# approx. 400x faster than og method, 2-3s total --> 0.005s total w/ NJIT, 8 cores
@njit(parallel=True)
def computeChernGridV2_vectorized(grid, thetaNUM):
    accumulator = np.zeros(NUM_STATES)  # Accumulator for each state

    for indexONE in prange(thetaNUM - 1):
        for indexTWO in range(thetaNUM - 1):
            # Extract eigenvectors for all states at once
            currentEigv00 = grid[indexONE, indexTWO]  # Shape: (num_components, num_states)
            currentEigv10 = grid[indexONE + 1, indexTWO]
            currentEigv01 = grid[indexONE, indexTWO + 1]
            currentEigv11 = grid[indexONE + 1, indexTWO + 1]

            # Compute inner products for all states simultaneously, sum over components-axis
            innerProductOne = np.sum(np.conj(currentEigv00) * currentEigv10, axis=0)
            innerProductTwo = np.sum(np.conj(currentEigv10) * currentEigv11, axis=0)
            innerProductThree = np.sum(np.conj(currentEigv11) * currentEigv01, axis=0)
            innerProductFour = np.sum(np.conj(currentEigv01) * currentEigv00, axis=0)

            # Compute Berry phase for all states concurrently
                # standard log approach
            # berryPhase = np.log(innerProductOne*innerProductTwo*innerProductThree*innerProductFour).imag

                # angle method is slightly better (?)
            # berryPhase = (
            #     np.angle(innerProductOne)
            #     + np.angle(innerProductTwo)
            #     + np.angle(innerProductThree)
            #     + np.angle(innerProductFour)
            # )
            berryPhase = np.angle(innerProductOne*innerProductTwo*innerProductThree*innerProductFour)

                # Phase shift to range [-π, π] (dont need this with np.angle constraint!)
            # berryPhase = (berryPhase + PI) % (2*PI) - PI

            # Accumulate Berry phases for all states
            accumulator += berryPhase
    return (accumulator / (2*PI)) + 1/NUM_STATES

# a fully vectorized approach for computing chern numbers, good for small systems, but slows down for N>64
def computeChernGridV2_fully_vectorized(grid, thetaNUM):
    # Stack the grid into a 4D array: (thetaNUM, thetaNUM, NumItems, NumStates)
    # stacked_grid = np.array([[grid[i, j] for j in range(thetaNUM)] for i in range(thetaNUM)])
    # stacked_grid.shape == (thetaNUM, thetaNUM, NumItems, NumStates)

    # Extract relevant neighbors using slicing
    Eigv00 = grid[:-1, :-1]  # Top-left corner
    Eigv10 = grid[1:, :-1]   # Bottom-left corner
    Eigv01 = grid[:-1, 1:]   # Top-right corner
    Eigv11 = grid[1:, 1:]    # Bottom-right corner

    # Compute inner products for all states and grid points
    innerProductOne = np.sum(np.conj(Eigv00) * Eigv10, axis=2)  # Sum over NumItems
    innerProductTwo = np.sum(np.conj(Eigv10) * Eigv11, axis=2)
    innerProductThree = np.sum(np.conj(Eigv11) * Eigv01, axis=2)
    innerProductFour = np.sum(np.conj(Eigv01) * Eigv00, axis=2)

    # Compute Berry phase contributions for all states and grid points
    berryPhase = np.angle(innerProductOne * innerProductTwo * innerProductThree * innerProductFour)

    # Sum over all grid points for each state
    accumulator = np.sum(berryPhase, axis=(0, 1))  # Sum over the grid dimensions (theta grid)

    # Normalize and compute Chern numbers
    return (accumulator / (2 * np.pi)) + 1 / NUM_STATES

def ensembleRun(n_iters,numTheta,csv_file):
    # Open the CSV file in write mode
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        if file.tell() == 0:  # Check if the file is empty
            writer.writerow(["State 1", "State 2", "State 3", "State 4", "State 5", "State 6", "State 7", "State 8"])

        # Loop to generate and write arrays
        for i in range(n_iters):  # Adjust the range for the desired number of rows
            chern_numbers = fullSimulationGrid(numTheta,visualize=False)
            writer.writerow(chern_numbers)
            file.flush()  
            print("ITERATION ", i, "COMPLETE")

# function to help time something like the hamiltonian construction
def timing():
    NUM=10
    # time = timeit.timeit("constructHamiltonian(256,3,7,V)", 'from __main__ import constructHamiltonian, PI, V', number=NUM)
    # time = timeit.timeit("optimizedConstructHamiltonian(16,8,2,V)", 'from __main__ import optimizedConstructHamiltonian, V', number=NUM)
    time = timeit.timeit("fullSimulationGrid(NUM_THETA,visualize=False)", 'from __main__ import fullSimulationGrid, NUM_THETA', number=NUM)
    
    print(f"Execution time: {time} seconds")
    print("Time per Call:", time/NUM)


if __name__ == "__main__":
    fullSimulationGrid(NUM_THETA,visualize=False)

    # V = constructPotential(POTENTIAL_MATRIX_SIZE)
    # constructHamiltonian(256,3,7,V)
    timing()

    # Comment the following lines for running an ensemble!
    # csv_file = "/scratch/gpfs/ed5754/iqheFiles/output.csv"
    # csv_file = "output256.csv"
    # ensembleRun(100,NUM_THETA,csv_file)

    # Following lines are for visualizing the V_mn potential strength, translated back to real space!
    
    # pot = constructPotential(POTENTIAL_MATRIX_SIZE)
    # pot = constructScatteringPotentialv2(POTENTIAL_MATRIX_SIZE)
    # helpers.plotRandomPotential(pot)
    # helpers.plotRandomPotential3D(pot)

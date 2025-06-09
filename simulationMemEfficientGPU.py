import numpy as np
import csv
import timeit
from numba import njit, prange, jit
from timeit import default_timer as timer
import os
from datetime import datetime
import cupy as cp
from collections import OrderedDict

import precompute
import helpers

# Define system-wide parameters...
# num-states is defined according to the ratio  
# N= B*L^2 / phi_o for phi_o=hc/e OR
# N = A / (2pi ell^2)=> so L^2 = 2pi*ell^2*N
# for ell^2 = hbar c / (e*B)

IMAG = 1j
PI = np.pi

NUM_STATES = 1024    # Number of states for the system
NUM_THETA=84        # Number of theta for the THETA x,y mesh
POTENTIAL_MATRIX_SIZE = int(4*np.sqrt(NUM_STATES))
# basically experimental parameter, complete more testing with this size... (keep L/M small)

# Precompute look-up tables (matrices) for Hamiltonian exponentials
mn_LUT, mj_LUT, mTheta_LUT = precompute.precompute_exponentialsNumPy(NUM_STATES, POTENTIAL_MATRIX_SIZE, NUM_THETA)
print("Precomputations Done")

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
@njit(parallel=True,fastmath=True)
def constructHamiltonianOriginal(matrix_size, theta_x, theta_y, matrixV):
    # define values as complex128, ie, double precision for real and imaginary parts
    H = np.zeros((matrix_size,matrix_size),dtype=np.complex128)

    # actually, we only require the lower-triangular portion of the matrix, since it's Hermitian, taking advantage of eigh functon!
    for j in prange(matrix_size):
        for k in range(j+1):
            # j,k entry in H
            H[j,k]=constructHamiltonianEntryOriginal(j,k,theta_x,theta_y,matrixV)
    return H

@njit()
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

    return V/np.sqrt(NUM_STATES)

# run a full iteration of the simulation. that is, compute the potential, then
# construct the hamiltonian and find the eigenvalues for a mesh of theta_x, theta_y's.
def fullSimulationGrid(thetaResolution=10,visualize=False):
    # for a given iteration, define a random potential...
    V = constructPotential(POTENTIAL_MATRIX_SIZE)

    print("Computing Chern Values...")
    cherns = computeChernGrid_gpu(thetaResolution,NUM_STATES,V)
    cherns = np.round(cherns,decimals=4)
    eigenvalues = saveEigenvalues(V)
    return V, eigenvalues, cherns

class EigenvectorCache:
    def __init__(self, max_bytes=64 * 1024**3, num_states=1024):  # 64GB max cache
        self.cache = OrderedDict()
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self.entry_size = num_states * num_states * 16  # complex128 size
        self.max_entries = max_bytes // self.entry_size

    def get(self, key):
        return self.cache.get(key, None)

    def add(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_entries:
            self.cache.popitem(last=False)  # Remove the oldest entry
        self.cache[key] = value

def computeChernGrid_gpu(thetaResolution, num_states, V):
    eig_cache = EigenvectorCache(max_bytes=64 * 1024**3, num_states=num_states)
    accumulator = np.zeros(num_states, dtype=np.float64)  # Move accumulation to CPU
    
    for indexONE in range(thetaResolution - 1):
        for indexTWO in range(thetaResolution - 1):
            
            keys = [(indexONE, indexTWO), (indexONE + 1, indexTWO),
                    (indexONE + 1, indexTWO + 1), (indexONE, indexTWO + 1)]
            eigVecs_cpu = []
            
            for key in keys:
                cached_eigv = eig_cache.get(key)
                if cached_eigv is None:
                    H_cpu = constructHamiltonian(num_states, key[0], key[1], V)  # Compute on CPU
                    H_gpu = cp.asarray(H_cpu)  # Transfer to GPU
                    _, eigv_gpu = cp.linalg.eigh(H_gpu)  # Compute eigenvectors on GPU
                    eigv_cpu = cp.asnumpy(eigv_gpu)  # Transfer back to CPU
                    eig_cache.add(key, eigv_cpu)
                else:
                    eigv_cpu = cached_eigv
                eigVecs_cpu.append(eigv_cpu)
            
            # Compute Berry phase on CPU
            innerProductOne = np.sum(np.conj(eigVecs_cpu[0]) * eigVecs_cpu[1], axis=0)
            innerProductTwo = np.sum(np.conj(eigVecs_cpu[1]) * eigVecs_cpu[2], axis=0)
            innerProductThree = np.sum(np.conj(eigVecs_cpu[2]) * eigVecs_cpu[3], axis=0)
            innerProductFour = np.sum(np.conj(eigVecs_cpu[3]) * eigVecs_cpu[0], axis=0)
            
            berryPhase = np.angle(innerProductOne * innerProductTwo * innerProductThree * innerProductFour)
            accumulator += berryPhase

    return (accumulator / (2 * np.pi)) + 1 / num_states

def saveEigenvalues(matrixV):
    eigenvalues00 = np.zeros(NUM_STATES)
    eigenvalues0PI = np.zeros(NUM_STATES)
    eigenvaluesPI0 = np.zeros(NUM_STATES)
    eigenvaluesPIPI = np.zeros(NUM_STATES)

    H_gpu = cp.zeros((NUM_STATES, NUM_STATES), dtype=cp.complex128)

    H_00 = constructHamiltonianOriginal(NUM_STATES,0,0,matrixV)
    H_gpu = cp.asarray(H_00)
    eigs_gpu, _ = cp.linalg.eigh(H_gpu, UPLO="L")
    eigenvalues00 = cp.asnumpy(eigs_gpu)

    H_0Pi = constructHamiltonianOriginal(NUM_STATES,0,PI,matrixV)
    H_gpu = cp.asarray(H_0Pi)
    eigs_gpu, _ = cp.linalg.eigh(H_gpu, UPLO="L")
    eigenvalues0PI = cp.asnumpy(eigs_gpu)

    H_Pi0 = constructHamiltonianOriginal(NUM_STATES,PI,0,matrixV)
    H_gpu = cp.asarray(H_Pi0)
    eigs_gpu, _ = cp.linalg.eigh(H_gpu, UPLO="L")    
    eigenvaluesPI0 = cp.asnumpy(eigs_gpu)

    H_PIPI = constructHamiltonianOriginal(NUM_STATES,PI,PI,matrixV)
    H_gpu = cp.asarray(H_PIPI)
    eigs_gpu, _ = cp.linalg.eigh(H_gpu, UPLO="L")       
    eigenvaluesPIPI = cp.asnumpy(eigs_gpu)

    return eigenvalues00, eigenvalues0PI, eigenvaluesPI0, eigenvaluesPIPI


def ensembleSave(n_iters,numTheta,dir):
    # Loop to generate and write arrays
    for iter in range(1,n_iters):  # Adjust the range for the desired number of rows
        V, eigenvalues, cherns = fullSimulationGrid(numTheta,visualize=False)
        save_trial_data(iter, V, cherns, eigenvalues, dir)
        print("ITERATION ", iter, "COMPLETE")

def save_trial_data(trial_num, V, chern_numbers, eigenvalues, output_dir):
    """Save all trial data in a single compressed .npz file efficiently."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate a timestamp in YYYY-MM-DD_HH-MM-SS format
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create the filename with the timestamp
    file_path = os.path.join(output_dir, f"trial_data_{trial_num}_{timestamp}.npz")
    # file_path = os.path.join(output_dir, f"trial_data_{trial_num}.npz")

    # Save all arrays efficiently using np.savez_compressed
    print("Saving Trial")
    np.savez_compressed(
        file_path,
        PotentialMatrix=V, 
        ChernNumbers=chern_numbers, 
        SumChernNumbers=np.sum(chern_numbers), 
        eigs00=eigenvalues[0], 
        eigs0pi=eigenvalues[1], 
        eigsPi0=eigenvalues[2], 
        eigsPipi=eigenvalues[3]
    )

if __name__ == "__main__":
    directory = "/scratch/gpfs/ed5754/iqheFiles/SaveData/N=1024_Mem/"
    ensembleSave(10000000,NUM_THETA,directory)
    # dict = load_trial_data('/Users/eddiedeleu/Desktop/IQHE Simulation/Optimized/Saved/trial_data_1.npz')
    # V = dict.get("PotentialMatrix")
    # print(V)
    # print(dict.get("ChernNumbers"))
    # print(dict.get("eigs00"))

    # fullSimulationGrid(NUM_THETA,visualize=True)

    # V = constructPotential(POTENTIAL_MATRIX_SIZE)
    # constructHamiltonian(256,3,7,V)
    # timing()

    # Comment the following lines for running an ensemble!
    # csv_file = "/scratch/gpfs/ed5754/iqheFiles/output.csv"
    # csv_file = "Testing64.csv"
    # ensembleRun(1000,NUM_THETA,csv_file)

    # Following lines are for visualizing the V_mn potential strength, translated back to real space!
    
    # pot = constructPotential(POTENTIAL_MATRIX_SIZE)
    # pot = constructScatteringPotentialv2(POTENTIAL_MATRIX_SIZE)
    # helpers.plotRandomPotential(pot)
    # helpers.plotRandomPotential3D(pot)

    # NEW method for saving data as .npy!

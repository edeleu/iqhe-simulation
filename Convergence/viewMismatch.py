# File for plotting mismatch between different trials, that is, focusing on the specific eigenvalue surfaces that are
# too close together.

import numpy as np
import pandas as pd
import csv, os
from numba import njit, prange, jit
import matplotlib.pyplot as plt
import random
import helpers

# Define system-wide parameters...
# num-states is defined according to the ratio  
# N= B*L^2 / phi_o for phi_o=hc/e OR
# N = A / (2pi ell^2)=> so L^2 = 2pi*ell^2*N
# for ell^2 = hbar c / (e*B)

IMAG = 1j
PI = np.pi

NUM_STATES = 128     # Number of states for the system
# NUM_THETA=26        # Number of theta for the THETA x,y mesh
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

def helper(thetaRes, V):
    thetas = np.linspace(0, 2*np.pi, num=thetaRes, endpoint=True)
    eigValueGrid = np.zeros((thetaRes, thetaRes, NUM_STATES, NUM_STATES), dtype=complex)
    min_eigs = np.zeros((thetaRes*thetaRes, NUM_STATES-1))
    index=0

    for i in range(thetaRes):
        for j in range(thetaRes):
            H = constructHamiltonian(NUM_STATES, thetas[i], thetas[j], V)
            eigs, eigv = np.linalg.eigh(H)
            eigValueGrid[i, j] = eigv

            min_eigs[index] = np.diff(eigs)  # Minimum spacing at (0,0)
            index += 1
    
    pctile = np.percentile(min_eigs,0.1,axis=1) # calculate lowest 20th percentile
    eigspacing = np.min(pctile) # get the closest spaced surface 20th pctile only...
    print(eigspacing)

    # Compute Chern numbers
    chern_numbers = np.round(computeChernGridV2_vectorized(eigValueGrid, thetaRes), decimals=3)
    chern_sum = np.sum(chern_numbers)

    return chern_numbers, chern_sum


if __name__ == "__main__":
    # NUM_THETA =72
    # mismatch = [21,22]
    # V_loaded = np.load("potential_matrix_trial_176_Nstates_64.npy")
    # V_loaded = np.load("/Users/eddiedeleu/Desktop/ToProcess/8/potential_matrix_trial_22645_Nstates_8.npy")
    # mismatch = [1,2]

    # grid = fullSimulationGrid(V_loaded,NUM_THETA)
    # helpers.plotEigenvalueMeshHelper(grid,NUM_THETA,NUM_STATES)
    # helpers.plotSpecificEigenvalues(grid,NUM_THETA,NUM_STATES,mismatch)
    # helper(30,V_loaded)


    # # # alternatively, test convergence...
    V_loaded = np.load("/Users/eddiedeleu/Desktop/ToProcess/128/potential_matrix_trial_270_Nstates_128_task_0.npy")
    ref_cherns, _ = helper(100,V_loaded)
    prevs = None
    for res in range (88,100,1):
        chern_numbers, chern_sum = helper(res,V_loaded)
        if prevs is None:
            prevs = chern_numbers
        else:
            if np.allclose(ref_cherns,chern_numbers):
                print(f"For {res}, Cherns match reference resolution")
            else:
                if np.allclose(prevs,chern_numbers):
                    print(f"For {res}, Cherns match previous resolution")
                else: print(f"For {res}, Cherns do not match previous resolution")
                # print(f"Sum is: {chern_sum}") 
                # print(chern_numbers)
                print()
            prevs = chern_numbers

    # folderPath = "/Users/eddiedeleu/Desktop/ToProcess/32"
    # for file_name in sorted(os.listdir(folderPath)):
    #     if file_name.endswith(".npy"):
    #         print(file_name)
    #         file_name = os.path.join(folderPath, file_name)
    #         V_loaded = np.load(file_name)

    #         ref_cherns, _ = helper(90,V_loaded)
    #         prevs = None
    #         for res in range (50,90,1):
    #             chern_numbers, chern_sum = helper(res,V_loaded)
    #             if prevs is None:
    #                 prevs = chern_numbers
    #             else:
    #                 if np.allclose(ref_cherns,chern_numbers):
    #                     print(f"For {res}, Cherns match reference resolution")
    #                 else:
    #                     if np.allclose(prevs,chern_numbers):
    #                         print(f"For {res}, Cherns match previous resolution")
    #                     else: print(f"For {res}, Cherns do not match previous resolution")
    #                     # print(f"Sum is: {chern_sum}") 
    #                     # print(chern_numbers)
    #                     print()
    #                 prevs = chern_numbers

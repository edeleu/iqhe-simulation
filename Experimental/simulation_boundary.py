import numpy as np
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

NUM_STATES = 64     # Number of states for the system
NUM_THETA=26        # Number of theta for the THETA x,y mesh
POTENTIAL_MATRIX_SIZE = int(4*np.sqrt(NUM_STATES))
# basically experimental parameter, complete more testing with this size... (keep L/M small)


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

    return V/NUM_STATES


def fullSimulationBoundary(V,thetaResolution=50):
    # for a given iteration, define a random potential...
    # V = constructPotential(POTENTIAL_MATRIX_SIZE)

    # next, loop over theta_x theta_y
    # we only need to go over the boundary rectangle of theta's
    thetas = np.linspace(0, 2*PI, num=thetaResolution, endpoint=True)
    delTheta = thetas[1]
    print(delTheta)

    # Loop 1: Construct bottom section of square
    bottomEigvValues=np.zeros((thetaResolution, NUM_STATES,NUM_STATES),dtype=complex)
    # Loop 2: Construct right section of square
    rightEigvValues=np.zeros((thetaResolution, NUM_STATES,NUM_STATES),dtype=complex)
    #Loop 3: Construct Top section of square
    topEigvValues =np.zeros((thetaResolution, NUM_STATES,NUM_STATES),dtype=complex)
    #Loop 4: Construct Left section of square
    leftEigvValues =np.zeros((thetaResolution, NUM_STATES,NUM_STATES),dtype=complex)

    for theta_index in range(thetaResolution):
        theta_x = thetas[theta_index]
        theta_y = thetas[theta_index]
        #Step 1: bottom segment where theta_y=0
        H = constructHamiltonianOriginal(NUM_STATES, theta_x, 0, V) 
        eigs, eigv = np.linalg.eigh(H)
        bottomEigvValues[theta_index] = helpers.fix_eigenvector_phases(eigv)

        #Step 2: right segment where theta_x=2*PI
        H = constructHamiltonianOriginal(NUM_STATES,2*PI, theta_y, V) 
        eigs, eigv = np.linalg.eigh(H)
        rightEigvValues[theta_index] = helpers.fix_eigenvector_phases(eigv)

        #Step 3: top segment where theta_y=2*PI
        H = constructHamiltonianOriginal(NUM_STATES, theta_x, 2*PI, V) 
        eigs, eigv = np.linalg.eigh(H)
        topEigvValues[theta_index] = helpers.fix_eigenvector_phases(eigv)

        #Step 4: left segment where theta_x=0
        H = constructHamiltonianOriginal(NUM_STATES, 0, theta_y, V) 
        eigs, eigv = np.linalg.eigh(H)
        leftEigvValues[theta_index] = helpers.fix_eigenvector_phases(eigv)

    # now, compute the N chern numbers.... 
    cherns = computeChernBoundary_vectorized(bottomEigvValues, rightEigvValues, topEigvValues, leftEigvValues,thetaResolution).real
    cherns = np.round(cherns, decimals=2)
    print(cherns)
    cherns = np.round(cherns, decimals=0)
    print("Sum",sum(cherns))

    # print("V2:")
    # cherns = computeChernBoundary_vectorizedV2(bottomEigvValues, rightEigvValues, topEigvValues, leftEigvValues,thetaResolution).real
    # cherns = np.round(cherns, decimals=2)
    # print(cherns)
    # cherns = np.round(cherns, decimals=0)
    # print("Sum",sum(cherns))
    return 1

def computeChernBoundary_vectorized(bottom, right, top, left, thetaNUM):
    print("Bottom-Right Discontinuity:", np.angle(np.vdot(bottom[-1, :, 0], right[0, :, 0])))
    print("Right-Top Discontinuity:", np.angle(np.vdot(right[-1, :, 0], top[-1, :, 0])))
    print("Top-Left Discontinuity:", np.angle(np.vdot(top[0, :, 0], left[-1, :, 0])))
    print("Left-Bottom Discontinuity:", np.angle(np.vdot(left[0, :, 0], bottom[0, :, 0])))

    bottom = np.array([helpers.fix_eigenvector_phases(vec) for vec in bottom])
    right = np.array([helpers.fix_eigenvector_phases(vec) for vec in right])
    top = np.array([helpers.fix_eigenvector_phases(vec) for vec in top])
    left = np.array([helpers.fix_eigenvector_phases(vec) for vec in left])
    accumulator = np.zeros(NUM_STATES)  # Accumulator for each state

    # step 1: loop thru bottom states
    for index in range(thetaNUM - 1):
        # Extract eigenvectors for all states at once
        currentEigv00 = bottom[index]  # Shape: (num_components, num_states)
        currentEigv10 = bottom[index + 1]

        # Compute inner products for all states simultaneously, sum over components-axis
        innerProductOne = np.sum(np.conj(currentEigv00) * currentEigv10, axis=0)
        phase = np.angle(innerProductOne)

        # Accumulate phases for all states around boundary
        accumulator += phase

    # step 2: loop thru right states
    for index in range(thetaNUM - 1):
        # Extract eigenvectors for all states at once
        currentEigv10 = right[index]  # Shape: (num_components, num_states)
        currentEigv11 = right[index + 1]

        # Compute inner products for all states simultaneously, sum over components-axis
        innerProductTwo = np.sum(np.conj(currentEigv10) * currentEigv11, axis=0)
        phase = np.angle(innerProductTwo)
        # Accumulate phases for all states around boundary
        accumulator += phase

    # # step 3: loop thru top states
    for index in range(thetaNUM-1, 0, -1):
        # Extract eigenvectors for all states at once
        currentEigv01 = top[index-1]  # Shape: (num_components, num_states)
        currentEigv11 = top[index]

        # Compute inner products for all states simultaneously, sum over components-axis
        innerProductThree = np.sum(np.conj(currentEigv11) * currentEigv01, axis=0)
        phase = np.angle(innerProductThree)
        # Accumulate phases for all states around boundary
        accumulator += phase

    # step 4: loop thru left states
    for index in range(thetaNUM-1, 0, -1):
        currentEigv01 = left[index]  # Shape: (num_components, num_states)
        currentEigv00 = left[index-1]

        # Compute inner products for all states simultaneously, sum over components-axis
        innerProductFour = np.sum(np.conj(currentEigv01) * currentEigv00, axis=0)
        phase = np.angle(innerProductFour)
        # print(phase[2])

        # Accumulate phases for all states around boundary
        accumulator += phase

    return ((accumulator / (2*PI)) + 1/NUM_STATES)

def computeChernBoundary_vectorizedV2(bottom, right, top, left, thetaNUM):
    print("Bottom-Right Discontinuity:", np.angle(np.vdot(bottom[-1, :, 0], right[0, :, 0])))
    print("Right-Top Discontinuity:", np.angle(np.vdot(right[-1, :, 0], top[-1, :, 0])))
    print("Top-Left Discontinuity:", np.angle(np.vdot(top[0, :, 0], left[-1, :, 0])))
    print("Left-Bottom Discontinuity:", np.angle(np.vdot(left[0, :, 0], bottom[0, :, 0])))

    bottom = np.array([helpers.fix_eigenvector_phases(vec) for vec in bottom])
    right = np.array([helpers.fix_eigenvector_phases(vec) for vec in right])
    top = np.array([helpers.fix_eigenvector_phases(vec) for vec in top])
    left = np.array([helpers.fix_eigenvector_phases(vec) for vec in left])
    accumulator = np.zeros((NUM_STATES),dtype=complex)  # Accumulator for each state

    # step 1: loop thru bottom states, list is 0 to 2pi
    for index in range(thetaNUM-1):
        # Extract eigenvectors for all states at once
        currentEigv00 = bottom[index]  # Shape: (num_components, num_states)
        currentEigv10 = bottom[(index + 1)%thetaNUM]

        # Compute inner products for all states simultaneously, sum over components-axis
        innerProductOne = np.sum(np.conj(currentEigv00) * currentEigv10, axis=0) -1
        # Accumulate phases for all states around boundary
        accumulator += innerProductOne

    # step 2: loop thru right states, list is 0 to 2pi
    for index in range(thetaNUM-1):
        # Extract eigenvectors for all states at once
        currentEigv10 = right[index]  # Shape: (num_components, num_states)
        currentEigv11 = right[(index + 1)%thetaNUM]

        # Compute inner products for all states simultaneously, sum over components-axis
        innerProductTwo = np.sum(np.conj(currentEigv10) * currentEigv11, axis=0)-1
        # Accumulate phases for all states around boundary
        accumulator += (innerProductTwo)

    # # step 3: loop thru top states, list is 0 to 2pi
    for index in range(thetaNUM-1):
        # Extract eigenvectors for all states at once
        currentEigv01 = top[index]  # Shape: (num_components, num_states)
        currentEigv11 = top[(index + 1)%thetaNUM]

        # Compute inner products for all states simultaneously, sum over components-axis
        innerProductThree = np.sum(np.conj(currentEigv01) * currentEigv11, axis=0)-1
        # Accumulate phases for all states around boundary
        accumulator -= innerProductThree 

    # step 4: loop thru left states, list is 0 to 2pi
    for index in range(thetaNUM-1):
        currentEigv01 = left[(index + 1)%thetaNUM]  # Shape: (num_components, num_states)
        currentEigv00 = left[index]

        # Compute inner products for all states simultaneously, sum over components-axis
        innerProductFour = np.sum(np.conj(currentEigv00) * currentEigv01, axis=0)-1
        # print(phase[2])

        # Accumulate phases for all states around boundary
        accumulator -= innerProductFour

    return (accumulator / (2*PI*IMAG))+1/NUM_STATES


# run a full iteration of the simulation. that is, compute the potential, then
# construct the hamiltonian and find the eigenvalues for a mesh of theta_x, theta_y's.
def fullSimulationGrid(V, thetaResolution=10,visualize=False):
    # for a given iteration, define a random potential...
    # V = constructPotential(POTENTIAL_MATRIX_SIZE)
    # V = constructScatteringPotential(POTENTIAL_MATRIX_SIZE)
    # V = constructScatteringPotentialv2(POTENTIAL_MATRIX_SIZE)

    # next, loop over theta_x theta_y (actually just the indices)
    thetas = np.linspace(0, 2*PI, num=thetaResolution, endpoint=True)
    delTheta = thetas[1]
    # print(delTheta)

    eigGrid = np.zeros((thetaResolution,thetaResolution),dtype=object)
    eigValueGrid = np.zeros((thetaResolution,thetaResolution, NUM_STATES,NUM_STATES),dtype=complex)
    for indexONE in range(thetaResolution):
        for indexTWO in range(thetaResolution):
            # H = constructHamiltonian(NUM_STATES,thetas[indexONE],thetas[indexTWO], V) 
            H = constructHamiltonianOriginal(NUM_STATES,thetas[indexONE],thetas[indexTWO], V) 

            eigs, eigv = np.linalg.eigh(H,UPLO="L")
            # eigs, eigv = scipy.linalg.eigh(H,driver="evd")

            # eigGrid[indexONE,indexTWO]=[thetas[indexONE],thetas[indexTWO],eigs]
            eigGrid[indexONE,indexTWO]=[thetas[indexONE],thetas[indexTWO],eigs]   
            eigValueGrid[indexONE,indexTWO]=eigv

    # standard method of computing chern-numbers individually
    # cherns=[]
    # for i in range(NUM_STATES):
    #     chernNum = computeChernGridV2(i,eigValueGrid,thetaResolution)
    #     # print(chernNum)
    #     # if np.abs(chernNum.imag) < 1e-8:
    #     #     chernNum = chernNum.real
    #     cherns.append(chernNum)
    # cherns = np.round(cherns,decimals=3)
    # print(cherns)
    # print("Sum",sum(cherns))

    # now, vectorized computation of chern-numbers!

    cherns = np.round(computeChernGridV2_vectorized(eigValueGrid,thetaResolution),decimals=3)

    if visualize:
        print(cherns)
        print("Sum",sum(cherns))
        # helpers.plotEigenvalueMeshHelper(eigGrid,thetaResolution,NUM_STATES)
    
    bottomEigvValues = eigValueGrid[:,0]
    rightEigvValues = eigValueGrid[-1,:]
    topEigvValues = eigValueGrid[:,-1]
    leftEigvValues = eigValueGrid[0,:]
    cherns = computeChernBoundary_vectorized(bottomEigvValues, rightEigvValues, topEigvValues, leftEigvValues,thetaResolution)
    cherns = np.round(cherns, decimals=3)
    print(cherns)
    cherns = np.round(cherns, decimals=0)
    print("Sum",sum(cherns))

    return cherns

#NJIT speedup only occurs when using ensemble, otherwise it actually slows down on the first pass...
# approx. 400x faster than og method, 2-3s total --> 0.07s total w/ NJIT, 8 cores
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


def saveEigenvalues(matrixV):
    eigenvalues00 = np.zeros(NUM_STATES)
    eigenvalues0PI = np.zeros(NUM_STATES)
    eigenvaluesPI0 = np.zeros(NUM_STATES)
    eigenvaluesPIPI = np.zeros(NUM_STATES)

    H_00 = constructHamiltonianOriginal(NUM_STATES,0,0,matrixV)
    eigs, _ = np.linalg.eigh(H_00,UPLO="L")
    eigenvalues00 = eigs

    H_0Pi = constructHamiltonianOriginal(NUM_STATES,0,0,matrixV)
    eigs, _ = np.linalg.eigh(H_0Pi,UPLO="L")
    eigenvalues0PI = eigs

    H_Pi0 = constructHamiltonianOriginal(NUM_STATES,0,0,matrixV)
    eigs, _ = np.linalg.eigh(H_Pi0,UPLO="L")
    eigenvaluesPI0 = eigs

    H_PIPI = constructHamiltonianOriginal(NUM_STATES,0,0,matrixV)
    eigs, _ = np.linalg.eigh(H_PIPI,UPLO="L")
    eigenvaluesPIPI = eigs

    return eigenvalues00, eigenvalues0PI, eigenvaluesPI0, eigenvaluesPIPI

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
    # fullSimulationBoundary(100)

    V = constructPotential(POTENTIAL_MATRIX_SIZE)
    # fullSimulationGrid(V, 26,visualize=True)
    fullSimulationBoundary(V, 100)
    fullSimulationBoundary(V, 200)
    fullSimulationBoundary(V, 400)
    fullSimulationBoundary(V, 800)

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

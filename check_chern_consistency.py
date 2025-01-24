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

# alternate way to construct potential (experimental)
def constructScatteringPotentialv1(size):
    # first pull numScatterers random xy locations and intensities
    VReal = np.zeros((2*size+1, 2*size+1), dtype=complex)
    for i in range(2*size+1):
        for j in range(2*size+1):
            VReal[i,j]=np.random.normal(0, 1)

    fourier_transformed = np.fft.fft2(VReal)
    # Shift the zero-frequency component to the center of the matrix
    centered_fourier = np.fft.fftshift(fourier_transformed)

    return centered_fourier

# method of constructing potential as in 2019 paper
def constructScatteringPotentialv2(size):
    totalScatterers = 16*NUM_STATES
    # first pull numScatterers random xy locations and intensities
    L = np.sqrt(2*PI*NUM_STATES) # take ell^2 =1
    deltaQ = 2*PI/L
    LSquared = L**2

    x_positions = np.random.uniform(0, L, totalScatterers)
    y_positions = np.random.uniform(0, L, totalScatterers)
    intensities = np.random.choice([-1,1],size=totalScatterers)

    V = np.zeros((2*size+1, 2*size+1), dtype=complex)

    # Generate Each V_mn in the entire grid
    for m in range(size + 1):
        for n in range(-size, size + 1):
            #we must now take the sum over all scatterers...
            accumulator = 0
            for imp in range(totalScatterers):
                x = x_positions[imp]
                y = y_positions[imp]
                accumulator += intensities[imp]*np.exp(-IMAG*deltaQ*(m*x+n*y))

            accumulator = accumulator/LSquared #??????
            V[size + m, size + n] = accumulator
            V[size - m, size - n] = np.conjugate(accumulator)
    
    V[size,size]=0

    return V

# run a full iteration of the simulation. that is, compute the potential, then
# construct the hamiltonian and find the eigenvalues for a mesh of theta_x, theta_y's.
def fullSimulationGrid(thetaResolution=10,visualize=False):
    # for a given iteration, define a random potential...
    V = constructPotential(POTENTIAL_MATRIX_SIZE)

    # thetaResolution = 16
    # thetas = np.linspace(0, 2*PI, num=thetaResolution, endpoint=True)
    # eigGrid = np.zeros((thetaResolution,thetaResolution),dtype=object)
    # eigValueGrid = np.zeros((thetaResolution,thetaResolution, NUM_STATES,NUM_STATES),dtype=complex)
    # for indexONE in range(thetaResolution):
    #     for indexTWO in range(thetaResolution):
    #         H = constructHamiltonian(NUM_STATES,thetas[indexONE],thetas[indexTWO], V) 
    #         # H = constructHamiltonian(NUM_STATES,thetas,indexTWO, V) 

    #         eigs, eigv = np.linalg.eigh(H,UPLO="L")
    #         # eigs, eigv = scipy.linalg.eigh(H,driver="evd")

    #         # eigGrid[indexONE,indexTWO]=[thetas[indexONE],thetas[indexTWO],eigs]
    #         eigGrid[indexONE,indexTWO]=[thetas[indexONE],thetas[indexTWO],eigs]   
    #         eigValueGrid[indexONE,indexTWO]=eigv

    # # now, vectorized computation of chern-numbers!

    # cherns16 = np.round(computeChernGridV2_vectorized(eigValueGrid,thetaResolution),decimals=3)

    # print("Num-Theta=16")
    # print(cherns16)
    # print("Sum",sum(cherns16))

    thetaResolution = 26
    thetas = np.linspace(0, 2*PI, num=thetaResolution, endpoint=True)
    eigGrid = np.zeros((thetaResolution,thetaResolution),dtype=object)
    eigValueGrid = np.zeros((thetaResolution,thetaResolution, NUM_STATES,NUM_STATES),dtype=complex)
    for indexONE in range(thetaResolution):
        for indexTWO in range(thetaResolution):
            H = constructHamiltonian(NUM_STATES,thetas[indexONE],thetas[indexTWO], V) 
            # H = constructHamiltonian(NUM_STATES,thetas,indexTWO, V) 

            eigs, eigv = np.linalg.eigh(H,UPLO="L")
            # eigs, eigv = scipy.linalg.eigh(H,driver="evd")

            # eigGrid[indexONE,indexTWO]=[thetas[indexONE],thetas[indexTWO],eigs]
            eigGrid[indexONE,indexTWO]=[thetas[indexONE],thetas[indexTWO],eigs]   
            eigValueGrid[indexONE,indexTWO]=eigv

    # now, vectorized computation of chern-numbers!

    cherns26 = np.round(computeChernGridV2_vectorized(eigValueGrid,thetaResolution),decimals=3)

    print("Num-Theta=26")
    print(cherns26)
    print("Sum",sum(cherns26))

    thetaResolution = 40
    thetas = np.linspace(0, 2*PI, num=thetaResolution, endpoint=True)
    eigValueGrid = np.zeros((thetaResolution,thetaResolution, NUM_STATES,NUM_STATES),dtype=complex)
    for indexONE in range(thetaResolution):
        for indexTWO in range(thetaResolution):
            H = constructHamiltonian(NUM_STATES,thetas[indexONE],thetas[indexTWO], V) 
            # H = constructHamiltonian(NUM_STATES,thetas,indexTWO, V) 

            eigs, eigv = np.linalg.eigh(H,UPLO="L")
            # eigs, eigv = scipy.linalg.eigh(H,driver="evd")

            # eigGrid[indexONE,indexTWO]=[thetas[indexONE],thetas[indexTWO],eigs]
            eigValueGrid[indexONE,indexTWO]=eigv

    # now, vectorized computation of chern-numbers!

    cherns40 = np.round(computeChernGridV2_vectorized(eigValueGrid,thetaResolution),decimals=3)

    print("Num-Theta=40")
    print(cherns40)
    print("Sum",sum(cherns40))

    thetaResolution = 50
    thetas = np.linspace(0, 2*PI, num=thetaResolution, endpoint=True)
    eigValueGrid = np.zeros((thetaResolution,thetaResolution, NUM_STATES,NUM_STATES),dtype=complex)
    for indexONE in range(thetaResolution):
        for indexTWO in range(thetaResolution):
            H = constructHamiltonian(NUM_STATES,thetas[indexONE],thetas[indexTWO], V) 
            # H = constructHamiltonian(NUM_STATES,thetas,indexTWO, V) 

            eigs, eigv = np.linalg.eigh(H,UPLO="L")
            # eigs, eigv = scipy.linalg.eigh(H,driver="evd")

            # eigGrid[indexONE,indexTWO]=[thetas[indexONE],thetas[indexTWO],eigs]
            eigValueGrid[indexONE,indexTWO]=eigv

    # now, vectorized computation of chern-numbers!

    cherns50 = np.round(computeChernGridV2_vectorized(eigValueGrid,thetaResolution),decimals=3)

    print("Num-Theta=50")
    print(cherns50)
    print("Sum",sum(cherns50))


    print("26==50?",np.allclose(cherns50,cherns26))
    print("26==40?",np.allclose(cherns26,cherns40))
    print("40==50?",np.allclose(cherns40,cherns50))
    print()

    return cherns40

def computeChernGridV2(stateNumber,grid,thetaNUM):
    accumulator = 0

    for indexONE in range(thetaNUM-1):
        for indexTWO in range(thetaNUM-1):
            currentEigv00 = grid[indexONE,indexTWO][:,stateNumber]
            currentEigv10 = grid[(indexONE+1),indexTWO][:,stateNumber]
            currentEigv01 = grid[indexONE,(indexTWO+1)][:,stateNumber]
            currentEigv11 = grid[(indexONE+1),(indexTWO+1)][:,stateNumber]

            innerProductOne = np.vdot(currentEigv00,currentEigv10)
            innerProductTwo = np.vdot(currentEigv10,currentEigv11)
            innerProductThree = np.vdot(currentEigv11,currentEigv01)
            innerProductFour = np.vdot(currentEigv01,currentEigv00)

            # we don't need to normalize anything since we don't care about the real part of the log! 

                # Simple implementation, do this and throw away resulting imag part since we dont normalize (ok!)
            # berryPhase = -IMAG*np.log(innerProductOne*innerProductTwo*innerProductThree*innerProductFour)
            # berryPhase = np.log(innerProductOne*innerProductTwo*innerProductThree*innerProductFour).imag

                # alternatively use arctangent to get only the part we care about!
                # can break up or do entire thing... 

            # product = innerProductOne*innerProductTwo*innerProductThree*innerProductFour
            # berryPhase=np.arctan2(product.imag,product.real)
            berryPhase = np.arctan2(innerProductOne.imag,innerProductOne.real) + np.arctan2(innerProductTwo.imag,innerProductTwo.real) + np.arctan2(innerProductThree.imag,innerProductThree.real) + np.arctan2(innerProductFour.imag,innerProductFour.real)

            # phase-shift to range of [-PI,PI]
            while berryPhase>PI:
                # print(berryPhase)
                berryPhase -= 2*PI
            
            while berryPhase<-PI:
                # print(berryPhase)
                berryPhase += 2*PI

            accumulator += berryPhase

    return (accumulator/(2*PI)+1/NUM_STATES)

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

def ensembleRun(n_iters,numTheta,csv_file):
    # Open the CSV file in write mode
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        if file.tell() == 0:  # Check if the file is empty
            writer.writerow(["State 1", "State 2", "State 3", "State 4", "State 5", "State 6", "State 7", "State 8"])

        # Loop to generate and write arrays
        for i in range(n_iters):  # Adjust the range for the desired number of rows
            chern_numbers = fullSimulationGrid(numTheta,visualize=True)
            writer.writerow(chern_numbers)
            file.flush()  
            print("ITERATION ", i, "COMPLETE")


if __name__ == "__main__":
    fullSimulationGrid(NUM_THETA,visualize=True)

    # V = constructPotential(POTENTIAL_MATRIX_SIZE)
    # constructHamiltonian(256,3,7,V)
    # timing()

    # Comment the following lines for running an ensemble!
    # csv_file = "/scratch/gpfs/ed5754/iqheFiles/output.csv"
    csv_file = "Testing64.csv"
    ensembleRun(1000,NUM_THETA,csv_file)
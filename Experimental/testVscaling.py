import numpy as np
import csv
import timeit
from numba import njit, prange, jit
from timeit import default_timer as timer
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

import precompute
import helpers

# Define system-wide parameters...
# num-states is defined according to the ratio  
# N= B*L^2 / phi_o for phi_o=hc/e OR
# N = A / (2pi ell^2)=> so L^2 = 2pi*ell^2*N
# for ell^2 = hbar c / (e*B)

IMAG = 1j
PI = np.pi

NUM_STATES = 64    # Number of states for the system
NUM_THETA=30        # Number of theta for the THETA x,y mesh
POTENTIAL_MATRIX_SIZE = int(4*np.sqrt(NUM_STATES))
# basically experimental parameter, complete more testing with this size... (keep L/M small)

# a simplified "original" version of constructing the entries, a bit easier to understand what's going on than above
# furthermore, we pass explicit thetas instead of indices
@njit(parallel=True,fastmath=True)
def constructHamiltonianOriginal(matrix_size, theta_x, theta_y, matrixV,potSize):
    # define values as complex128, ie, double precision for real and imaginary parts
    H = np.zeros((matrix_size,matrix_size),dtype=np.complex128)

    # actually, we only require the lower-triangular portion of the matrix, since it's Hermitian, taking advantage of eigh functon!
    for j in prange(matrix_size):
        for k in range(j+1):
            # j,k entry in H
            H[j,k]=constructHamiltonianEntryOriginal(j,k,theta_x,theta_y,matrixV,potSize,matrix_size)
    return H

@njit()
def constructHamiltonianEntryOriginal(indexJ,indexK,theta_x,theta_y,matrixV,potSize,N):
    value = 0
    
    for n in range(-potSize, potSize+1):
        if deltaPBC(indexJ,indexK-n,N):
            for m in range(-potSize, potSize+1):
                potentialValue = matrixV[m+potSize,n+potSize]
                exp_term = (1/N)*((-PI/2)*(m**2+n**2)-IMAG*PI*m*n+IMAG*(m*theta_y-n*theta_x)-IMAG*m*2*PI*indexJ)
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
def deltaPBC(a,b,N):
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

    # return 50*V/(size*size)
    return V/np.sqrt(N)

def constructScatteringPotentialv2(size,N):
    totalScatterers = 16*N
    # first pull numScatterers random xy locations and intensities
    L = np.sqrt(2*PI*N) # take ell^2 =1
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


# N = 256    # Number of states for the system
# NUM_THETA=30        # Number of theta for the THETA x,y mesh
# POTENTIAL_MATRIX_SIZE = int(4*np.sqrt(NUM_STATES))
# # pot = constructPotential(144)
# pot = constructScatteringPotentialv2(POTENTIAL_MATRIX_SIZE,N)
# helpers.plotRandomPotential(pot)

# mini=[]
# maxi=[]
# lisT=range(8,512,64)
# for N in range(8,512,64):
#     pot_size = int(4*np.sqrt(N))
#     V = constructPotential(pot_size)
#     # V = constructScatteringPotentialv2(pot_size,N)

#     H = constructHamiltonianOriginal(N,PI,PI, V,pot_size) 

#     eigs, eigv = np.linalg.eigh(H,UPLO="L")
#     mini.append(min(eigs))
#     maxi.append(max(eigs))

#     print(N)
#     print("Min",min(eigs))
#     print("Max",max(eigs))
#     print("RMS", np.sqrt(np.sum(np.abs(V)**2)/(pot_size*pot_size)))
#     print()
# plt.plot(lisT,mini)
# plt.plot(lisT,maxi)

# eigsS = np.array([])
# N=1024
# pot_size = int(4*np.sqrt(N))
# for i in range(50):
#     V = constructPotential(pot_size)
#     H = constructHamiltonianOriginal(N,PI,PI, V,pot_size) 
#     eigs, eigv = np.linalg.eigh(H,UPLO="L")
#     eigsS = np.concatenate((eigsS, eigs))

eigsSS = np.array([])
N=32
pot_size = int(4*np.sqrt(N))
for i in range(5000):
    V = constructPotential(pot_size)
    H = constructHamiltonianOriginal(N,PI,PI, V,pot_size) 
    eigs, eigv = np.linalg.eigh(H,UPLO="L")
    eigsSS = np.concatenate((eigsSS, eigs))

# param = stats.semicircular.fit(eigsSS)
x = np.linspace(eigsSS.min(), eigsSS.max(), 100)
# pdf_fitted = stats.semicircular.pdf(x, *param)

# R=3
# pdf_fitted = (2 / (np.pi * R**2)) * np.sqrt(R**2 - x**2)
kde = stats.gaussian_kde(eigsSS, bw_method=0.2)

plt.hist(eigsSS, bins=100,density=True)
# plt.hist(eigsS, bins=50, alpha=0.6) #density=True, cumulative=True
plt.plot(x, kde(x), color='r')

plt.show()

import numpy as np
PI = np.pi
IMAG = 1j

# A helper function to precompute valid n-values to iterate through when constructing the Hamiltonian
# in accordance with the periodic boundary conditions
def precompute_n_PBC(N, size):
    lookup_table = {}

    for j in range(N):  # Loop through all values of j
        for k in range(N):  # Loop through all values of k
            valid_n = []  # List to store valid n for the current (j, k)

            for n in range(-size, size + 1):  # Loop through the range of n
                # Compute modulus and check if it equals 0... Delta function!
                modulus = (j - (k - n)) % N
                if modulus == 0:
                    valid_n.append(n)

            lookup_table[(j, k)] = valid_n  # Store the valid n values for (j, k)

    return lookup_table

# A helper function to precompute each 2-variable section of the Hamiltonian exponential
# ie, for mn, mj, mthetaY, nThetaX, and mj.
def precompute_exponentials(N, Vsize, numTheta):
    thetas = np.linspace(0, 2*PI, num=numTheta, endpoint=True)

    mn_LUT = {}
    for n in range(-Vsize, Vsize+1):
        for m in range(-Vsize, Vsize+1):
            mn_LUT[(m,n)] = np.exp(-PI*(m**2+n**2)/(2*N)-IMAG*PI*m*n/N)

    mj_LUT = {}
    for indexJ in range(N):
        for m in range(-Vsize, Vsize+1):
            mj_LUT[(m,indexJ)] = np.exp(-IMAG*2*PI*m*indexJ/N)

    mThetaY_LUT = {}
    for m in range(-Vsize, Vsize+1):
        for thetaIndex in range(numTheta):
            thetaY = thetas[thetaIndex]
            mThetaY_LUT[(m,thetaIndex)] = np.exp(IMAG*(m*thetaY)/N)

    nThetaX_LUT = {}
    for n in range(-Vsize, Vsize+1):
        for thetaIndex in range(numTheta):
            thetaX = thetas[thetaIndex]
            nThetaX_LUT[(n,thetaIndex)] = np.exp(-IMAG*(n*thetaX)/N)
    
    return mn_LUT, mThetaY_LUT, nThetaX_LUT, mj_LUT

# An updated helper function that computes 2 look-up tables for the Hamiltonian exponential
# for small systems, it's quicker to use mnj look-up table instead of two
# Furthermore, mTheta can be used for nTheta so both are not computed.
def precompute_exponentialsV2(N, Vsize, numTheta):
    thetas = np.linspace(0, 2*PI, num=numTheta, endpoint=True)

    mnj_LUT = {}
    for indexJ in range(N):
        for n in range(-Vsize, Vsize+1):
            for m in range(-Vsize, Vsize+1):
                mnj_LUT[(m,n,indexJ)] = np.exp(-PI*(m**2+n**2)/(2*N)-IMAG*PI*m*n/N-IMAG*2*PI*m*indexJ/N)

    mTheta_LUT = {}
    for m in range(-Vsize, Vsize+1):
        for thetaIndex in range(numTheta):
            theta = thetas[thetaIndex]
            mTheta_LUT[(m,thetaIndex)] = np.exp(IMAG*(m*theta)/N)
    
    return mnj_LUT, mTheta_LUT

# A different implementation to precompute the exponentials for the Hamiltonian, safe with Numba
# instead of using dictionary, use 3 NumPy matrices for lookup.
def precompute_exponentialsNumPy(N, Vsize, numTheta):
    thetas = np.linspace(0, 2*PI, num=numTheta, endpoint=True)
    
    mn_LUT =  np.zeros((2*Vsize+1,2*Vsize+1),dtype=np.complex128)
    for n in range(-Vsize, Vsize+1):
        for m in range(-Vsize, Vsize+1):
            mn_LUT[Vsize+m,Vsize+n] = np.exp(-PI*(m**2+n**2)/(2*N)-IMAG*PI*m*n/N)

    mj_LUT =  np.zeros((2*Vsize+1,N),dtype=np.complex128)
    for indexJ in range(N):
        for m in range(-Vsize, Vsize+1):
            mj_LUT[Vsize+m,indexJ] = np.exp(-IMAG*2*PI*m*indexJ/N)

    # MNJ is slower for >256 systems, don't use this
    # mnj_LUT = np.zeros((2*Vsize+1,2*Vsize+1,N),dtype=np.complex128)
    # for indexJ in range(N):
    #     for n in range(-Vsize, Vsize+1):
    #         for m in range(-Vsize, Vsize+1):
    #             mnj_LUT[m+Vsize,n+Vsize,indexJ] = np.exp(-PI*(m**2+n**2)/(2*N)-IMAG*PI*m*n/N-IMAG*2*PI*m*indexJ/N)
    #             # mnj_LUT[(m,n,indexJ)] = np.exp(-PI*(m**2+n**2)/(2*N)-IMAG*PI*m*n/N-IMAG*2*PI*m*indexJ/N)
    
    mTheta_LUT = np.zeros((2*Vsize+1,numTheta),dtype=np.complex128)
    for m in range(-Vsize, Vsize+1):
        for thetaIndex in range(numTheta):
            theta = thetas[thetaIndex]
            mTheta_LUT[m+Vsize,thetaIndex] = np.exp(IMAG*(m*theta)/N)
    return mn_LUT, mj_LUT, mTheta_LUT

# if __name__ == "__main__":
#     # Perform any testing here...
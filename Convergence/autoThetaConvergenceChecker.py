import numpy as np
import pandas as pd
import csv, os
from numba import njit, prange, jit
from timeit import default_timer as timer

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

def run_convergence_trials(max_resolution=50, min_resolution=10, step=2, num_trials=10, save_path="chern_convergence_results.csv"):
    """
    Runs multiple trials to test the convergence of computed Chern numbers 
    with respect to decreasing theta-grid resolution.

    Saves a spreadsheet with:
    - Trial number
    - Converged theta resolution (last correct matching resolution)
    - Minimum eigenvalue spacing at theta = (0,0)
    """
    results = []

    for trial in range(1, num_trials + 1):
        print(f"Starting Trial {trial}...")

        # Generate a fixed Potential matrix V for this trial
        V = constructPotential(POTENTIAL_MATRIX_SIZE)

        # Initialize reference values
        last_correct_resolution = max_resolution
        min_eigenvalue_spacing00 = None
        min_eigenvalue_spacingPIPI = None
        min_eigenvalue_spacing_reference = None
        min_eigenvalue_spacing_converged = None
        prev_min_spaced_eigs = None
        previous_correct_chern = None

        H = constructHamiltonian(NUM_STATES, 0, 0, V)
        eigs, eigv = np.linalg.eigh(H)
        min_eigenvalue_spacing00 = np.min(np.diff(eigs))  # Minimum spacing at (0,0)
        H = constructHamiltonian(NUM_STATES, PI, PI, V)
        eigs, eigv = np.linalg.eigh(H)
        min_eigenvalue_spacingPIPI = np.min(np.diff(eigs))  # Minimum spacing at (PI,PI)

        for thetaResolution in range(max_resolution, min_resolution - 1, -step):
            print(f"Computing for theta resolution {thetaResolution}...")

            # Compute eigenvalues & eigenvectors
            thetas = np.linspace(0, 2*np.pi, num=thetaResolution, endpoint=True)
            eigValueGrid = np.zeros((thetaResolution, thetaResolution, NUM_STATES, NUM_STATES), dtype=complex)
            theta_min_spacing = min_eigenvalue_spacing00 # just initialize it here
            minimum_ij_indices = [0,0]
            for i in range(thetaResolution):
                for j in range(thetaResolution):
                    H = constructHamiltonian(NUM_STATES, thetas[i], thetas[j], V)
                    eigs, eigv = np.linalg.eigh(H)
                    eigValueGrid[i, j] = eigv

                    min_eigenvalue_spacing = np.min(np.diff(eigs))  # Minimum spacing at (0,0)
                    if min_eigenvalue_spacing<theta_min_spacing:
                        theta_min_spacing = min_eigenvalue_spacing
                        min_spaced_eigs = np.diff(eigs)

            # Compute Chern numbers
            chern_numbers = np.round(computeChernGridV2_vectorized(eigValueGrid, thetaResolution), decimals=3)
            chern_sum = np.sum(chern_numbers)

            # Special handling for max resolution case
            if thetaResolution == max_resolution:
                if chern_sum == 1.0:
                    previous_correct_chern = chern_numbers
                    min_eigenvalue_spacing_reference = theta_min_spacing
                    print(f"Correct Chern-Numbers are at {thetaResolution}")
                    print(chern_numbers)
                else:
                    print(f"Warning: Max resolution {max_resolution} does not sum to 1. Checking next smaller resolution...")
                    continue  # Move to the next smaller resolution

            # If the previous max resolution didn't sum to 1, check if this one can be used as reference
            if previous_correct_chern is None:
                if chern_sum == 1.0:
                    previous_correct_chern = chern_numbers
                    min_eigenvalue_spacing_reference = theta_min_spacing
                    print(f"Correct Chern-Numbers are at {thetaResolution}")
                    print(chern_numbers)
                    continue  
                else:
                    print(f"Stopping early: No reliable reference resolution found.")
                    break  

            # If sum is not 1, terminate
            if chern_sum != 1.0:
                print(f"Warning: Chern numbers sum to {chern_sum} at resolution {thetaResolution}.")
                print(f"Previous minimum eigenvalue spacing, {prev_min_spaced_eigs}")
                print(f"Current minimum eigenvalue spacing, {min_spaced_eigs}")
                print()
                mismatches = highlight_mismatches(previous_correct_chern, chern_numbers)
                print(f"nMismatch Details: {mismatches}")
                break  

            # If this resolution matches the previous correct one, update last correct resolution
            if np.allclose(chern_numbers, previous_correct_chern):
                last_correct_resolution = thetaResolution
                min_eigenvalue_spacing_converged = theta_min_spacing
                anomaly_flag = False  # Reset anomaly flag since we have a valid match
                previous_correct_chern = chern_numbers  # Update reference for future comparisons
                prev_min_spaced_eigs = min_spaced_eigs
                continue  

            # Otherwise, raise the anomaly flag and allow one more step
            print(f"Warning: Mismatch detected at resolution {thetaResolution}.")
            mismatches = highlight_mismatches(previous_correct_chern, chern_numbers)
            print(f"nMismatch Details: {mismatches}")
            print()
            print(f"Previous minimum eigenvalue spacing, {prev_min_spaced_eigs}")
            print(f"Current minimum eigenvalue spacing, {min_spaced_eigs}")
            break

        save_to_csv(trial,last_correct_resolution,min_eigenvalue_spacing00, min_eigenvalue_spacingPIPI, min_eigenvalue_spacing_reference, min_eigenvalue_spacing_converged,save_path)
        print(f"Trial {trial} results saved to {save_path}")

def highlight_mismatches(reference, current):
    """Returns a formatted string with mismatched values marked in place.
    
    Mismatches will be highlighted with a '*' and the original value will be shown in parentheses.
    """
    formatted_output = []
    
    for ref_val, cur_val in zip(reference, current):
        # Check if values match
        if np.isclose(ref_val, cur_val):
            formatted_output.append(str(cur_val))  # Value matches, no special marker
        else:
            # Value mismatch, mark it with "*" and show the original reference value
            formatted_output.append(f"{cur_val}* [REF: {ref_val}]")
    
    # Join the formatted values into a single string for output
    return ", ".join(formatted_output)

def save_to_csv(trial_number, converged_resolution, min_eigenvalue_spacing00, min_eigenvalue_spacingPIPI, min_eigenvalue_spacing_reference, min_eigenvalue_spacing_converged, filename="chern_data.csv"):
    # Check if the file already exists
    file_exists = os.path.isfile(filename)

    # Prepare the data to be saved
    data = {
        "Trial Number": [trial_number],
        "Converged Resolution": [converged_resolution],
        "Min Eigenvalue Spacing 00": [min_eigenvalue_spacing00],
        "Min Eigenvalue Spacing PiPi": [min_eigenvalue_spacingPIPI],
        "Min Eigenvalue Spacing Reference": [min_eigenvalue_spacing_reference],
        "Min Eigenvalue Spacing Converged": [min_eigenvalue_spacing_converged],
    }
    
    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Append data to the CSV, including headers only if the file doesn't exist
    df.to_csv(filename, mode='a', header=not file_exists, index=False)
    
    print(f"Data saved to {filename}. Trial {trial_number}: Resolution {converged_resolution}, Min Eigenvalue Reference {min_eigenvalue_spacing_reference},Min Eigenvalue Converged {min_eigenvalue_spacing_converged}")

if __name__ == "__main__":
    # Run the convergence test
    run_convergence_trials()

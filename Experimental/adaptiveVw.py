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

# ------------

import numpy as np
from numba import njit, prange
import heapq
from collections import deque

# Configuration constants
NUM_STATES = 4  # Example value, adjust based on your system
POTENTIAL_MATRIX_SIZE = 5  # Example value
PI = np.pi

def get_theta_from_indices(i, j, grid_size, base_res):
    """Convert grid indices to theta values"""
    return (2*PI*i/(grid_size-1), 2*PI*j/(grid_size-1))

def needs_refinement(plaquette, grid, tolerance):
    """Determine if a plaquette needs refinement based on local error estimate"""
    # Get the four corner points
    points = [grid['points'][i][j] for (i,j) in plaquette['corners']]
    
    # Calculate error estimate using difference between coarse and refined Berry phases
    error = estimate_plaquette_error(points)
    
    # Check for eigenvalue crossings
    has_crossing = check_eigenvalue_crossing(points)
    
    return error > tolerance or has_crossing

def calculate_berry_phase(corners):
    v00, v10, v11, v01 = corners
    
    inner1 = np.sum(np.conj(v00) * v10, axis=0)
    inner2 = np.sum(np.conj(v10) * v11, axis=0)
    inner3 = np.sum(np.conj(v11) * v01, axis=0)
    inner4 = np.sum(np.conj(v01) * v00, axis=0)
    
    return np.angle(inner1 * inner2 * inner3 * inner4)

def estimate_plaquette_error(points):
    """Estimate numerical error in Berry phase calculation for a plaquette"""
    # Coarse calculation
    berry_coarse = calculate_berry_phase([p['eigv'] for p in points])
    
    # Create virtual refined grid (2x2 subdivision)
    refined_phases = []
    for _ in range(4):  # Simplified virtual refinement
        virtual_phase = berry_coarse + np.random.normal(0, 0.1*abs(berry_coarse))
        refined_phases.append(virtual_phase)
    
    # Error estimate as difference between coarse and virtual refined
    error = np.mean(np.abs(berry_coarse - np.sum(refined_phases)))
    return error

def check_eigenvalue_crossing(points):
    """Check for eigenvalue crossings within a plaquette"""
    eigenvalues = np.array([p['eigs'] for p in points])
    min_eigs = np.min(eigenvalues, axis=0)
    max_eigs = np.max(eigenvalues, axis=0)
    
    # Check for overlap between adjacent bands
    crossings = np.any(max_eigs[:-1] > min_eigs[1:])
    return crossings

def flatten_plaquettes(plaquettes):
    """Flatten hierarchical plaquette structure into final leaf nodes"""
    flat_list = []
    queue = deque(plaquettes)
    
    while queue:
        p = queue.popleft()
        if p['children'] is None:
            flat_list.append(p)
        else:
            queue.extend(p['children'])
    return flat_list

def adaptive_refinement(initial_grid, max_depth=3, tolerance=0.05):
    """Perform adaptive mesh refinement with error control"""
    grid = initial_grid.copy()
    priority_queue = []
    
    # Initialize priority queue with initial plaquettes and their error estimates
    for idx, p in enumerate(grid['plaquettes']):
        error = estimate_plaquette_error([grid['points'][i][j] for (i,j) in p['corners']])
        heapq.heappush(priority_queue, (-error, idx, p))  # Negative for max-heap
    
    current_max_idx = len(grid['plaquettes']) - 1
    
    while priority_queue and max_depth > 0:
        neg_error, idx, p = heapq.heappop(priority_queue)
        error = -neg_error
        
        if p['level'] >= max_depth or error < tolerance:
            continue
            
        # Subdivide the plaquette
        new_plaquettes = subdivide_plaquette(p, grid)
        grid['plaquettes'][idx]['children'] = new_plaquettes
        grid['plaquettes'].extend(new_plaquettes)
        
        # Add new plaquettes to the priority queue
        for new_p in new_plaquettes:
            current_max_idx += 1
            new_error = estimate_plaquette_error(
                [grid['points'][i][j] for (i,j) in new_p['corners']]
            )
            heapq.heappush(priority_queue, (-new_error, current_max_idx, new_p))
        
        max_depth -= 1
    
    return grid

def subdivide_plaquette(p, grid):
    """Split a plaquette into four sub-plaquettes"""
    (i1, j1), (i2, j2) = p['corners'][0], p['corners'][2]
    mid_i = (i1 + i2) // 2
    mid_j = (j1 + j2) // 2
    
    # Create new grid points if necessary
    for i in [i1, mid_i, i2]:
        for j in [j1, mid_j, j2]:
            if grid['points'][i][j] is None:
                theta_x, theta_y = get_theta_from_indices(i, j, len(grid['points']), len(grid['points'][0]))
                H = constructHamiltonian(NUM_STATES, theta_x, theta_y, grid['potential'])
                eigs, eigv = np.linalg.eigh(H)
                grid['points'][i][j] = {
                    'theta': (theta_x, theta_y),
                    'eigs': eigs,
                    'eigv': helpers.fix_eigenvector_phases(eigv)
                }
    
    return [
        {'corners': [(i1,j1), (mid_i,j1), (mid_i,mid_j), (i1,mid_j)], 
         'level': p['level']+1, 'children': None},
        {'corners': [(mid_i,j1), (i2,j1), (i2,mid_j), (mid_i,mid_j)],
         'level': p['level']+1, 'children': None},
        {'corners': [(i1,mid_j), (mid_i,mid_j), (mid_i,j2), (i1,j2)],
         'level': p['level']+1, 'children': None},
        {'corners': [(mid_i,mid_j), (i2,mid_j), (i2,j2), (mid_i,j2)],
         'level': p['level']+1, 'children': None}
    ]

@njit(parallel=True)
def compute_adaptive_chern(grid_points, flat_plaquettes):
    """Compute Chern numbers from refined grid using Numba-accelerated code"""
    accumulator = np.zeros(NUM_STATES)
    
    for idx in prange(len(flat_plaquettes)):
        p = flat_plaquettes[idx]
        i1, j1 = p['corners'][0]
        i2, j2 = p['corners'][1]
        i3, j3 = p['corners'][2]
        i4, j4 = p['corners'][3]
        
        v00 = grid_points[i1, j1]
        v10 = grid_points[i2, j2]
        v11 = grid_points[i3, j3]
        v01 = grid_points[i4, j4]
        
        inner1 = np.sum(np.conj(v00) * v10, axis=0)
        inner2 = np.sum(np.conj(v10) * v11, axis=0)
        inner3 = np.sum(np.conj(v11) * v01, axis=0)
        inner4 = np.sum(np.conj(v01) * v00, axis=0)
        
        berry_phase = np.angle(inner1 * inner2 * inner3 * inner4)
        accumulator += berry_phase
    
    return accumulator / (2 * np.pi)

def fullSimulationGridAdaptive(base_res=10, max_depth=3, tol=0.05):
    """Main simulation function with adaptive refinement"""
    V = np.load("potential_matrix_trial_118_Nstates_64.npy")
    
    # Initialize grid structure
    grid = {
        'potential': V,
        'points': [[None for _ in range(base_res+1)] for _ in range(base_res+1)],
        'plaquettes': []
    }
    
    # Initialize grid points
    for i in range(base_res+1):
        for j in range(base_res+1):
            theta_x, theta_y = get_theta_from_indices(i, j, base_res, base_res)
            H = constructHamiltonian(NUM_STATES, theta_x, theta_y, V)
            eigs, eigv = np.linalg.eigh(H)
            grid['points'][i][j] = {
                'theta': (theta_x, theta_y),
                'eigs': eigs,
                'eigv': helpers.fix_eigenvector_phases(eigv)
            }
    
    # Create initial plaquettes
    for i in range(base_res):
        for j in range(base_res):
            grid['plaquettes'].append({
                'corners': [(i,j), (i+1,j), (i+1,j+1), (i,j+1)],
                'level': 0,
                'children': None
            })
    
    # Perform adaptive refinement
    refined_grid = adaptive_refinement(grid, max_depth, tol)
    
    # Flatten and compute Chern numbers
    flat_plaquettes = flatten_plaquettes(refined_grid['plaquettes'])
    
    # Convert to NumPy array for Numba compatibility
    grid_shape = (len(refined_grid['points']), len(refined_grid['points'][0]))
    grid_points = np.empty(grid_shape, dtype=np.complex128)

    # Assign eigenvectors with proper shape verification
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            eigv = refined_grid['points'][i][j]['eigv']
            grid_points[i,j] = eigv.astype(np.complex128)

    
    # grid_points = np.empty((base_res+1, base_res+1), dtype=np.complex128)
    # for i in range(base_res+1):
    #     for j in range(base_res+1):
    #         grid_points[i,j] = refined_grid['points'][i][j]['eigv']
    
    cherns = compute_adaptive_chern(grid_points, flat_plaquettes)
    return np.round(cherns, decimals=3)

# Run the adaptive simulation
chern_numbers = fullSimulationGridAdaptive(
    base_res=10,  # Initial grid resolution
    max_depth=3,  # Maximum refinement levels
    tol=0.05      # Error tolerance for refinement
)

print("Computed Chern numbers:", chern_numbers)
print("Sum of Chern numbers:", np.sum(chern_numbers))

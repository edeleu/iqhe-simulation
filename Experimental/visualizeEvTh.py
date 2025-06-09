import numpy as np
import csv
import timeit
# from numba import njit, prange, jit
from timeit import default_timer as timer
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Plot settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "lines.linewidth": 1,
    "grid.alpha": 0.3,
    "axes.grid": True,
})

# Define system-wide parameters...
# num-states is defined according to the ratio  
# N= B*L^2 / phi_o for phi_o=hc/e OR
# N = A / (2pi ell^2)=> so L^2 = 2pi*ell^2*N
# for ell^2 = hbar c / (e*B)

IMAG = 1j
PI = np.pi

NUM_STATES = 8    # Number of states for the system
NUM_THETA=35        # Number of theta for the THETA x,y mesh
POTENTIAL_MATRIX_SIZE = int(4*np.sqrt(NUM_STATES))

def constructHamiltonianOriginal(matrix_size, theta_x, theta_y, matrixV):
    # define values as complex128, ie, double precision for real and imaginary parts
    H = np.zeros((matrix_size,matrix_size),dtype=np.complex128)

    # actually, we only require the lower-triangular portion of the matrix, since it's Hermitian, taking advantage of eigh functon!
    for j in range(matrix_size):
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

def plotEigenvalueMeshHelper(grid, numTheta, N, ax):
    colors = [
    "#66c2a5",  # soft teal
    "#fc8d62",  # soft orange
    "#8da0cb",  # soft blue
    "#e78ac3",  # soft pink
    "#a6d854",  # lime green
    "#ba3c3c",  # muted red
    "#b3b3b3",  # soft gray
    "#7570b3",  # bold purplish blue (new, replaces yellow)
]
    
    # Plot a surface for each state
    for idx, color in enumerate(colors): 
        X = np.array([[grid[i, j][0] for j in range(numTheta)] for i in range(numTheta)])
        Y = np.array([[grid[i, j][1] for j in range(numTheta)] for i in range(numTheta)])
        Z = np.array([[grid[i, j][2][idx] for j in range(numTheta)] for i in range(numTheta)])

        # Plot the surface with optimized viewing settings
        ax.plot_surface(X, Y, Z, color=color, edgecolor='none', alpha=0.85)

    # Set labels
    ax.set_xlabel(r'$\theta_x$', labelpad=-13)
    ax.set_ylabel(r'$\theta_y$', labelpad=-13)
    ax.set_zlabel(r'$E$', labelpad=-11)

    # Customize tick marks with LaTeX formatting
    ticks = [0, PI/2, PI, 3*PI/2, 2*PI]
    tick_labels = [r"$0$", "", "", "", r"$2\pi$"]
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    ax.view_init(elev=13., azim=-52)
    ax.tick_params(axis='x', which='major', pad=-6)
    ax.tick_params(axis='y', which='major', pad=-6)
    ax.tick_params(axis='z', which='major', pad=-4)
    ax.set_zlim(-2.5,2.5)

def sim(thetaResolution):
    V = constructPotential(POTENTIAL_MATRIX_SIZE)

    thetas = np.linspace(0, 2*PI, num=thetaResolution, endpoint=True)
    eigGrid = np.zeros((thetaResolution, thetaResolution), dtype=object)
    
    for indexONE in range(thetaResolution):
        for indexTWO in range(thetaResolution):
            H = constructHamiltonianOriginal(NUM_STATES, thetas[indexONE], thetas[indexTWO], V) 

            eigs, eigv = np.linalg.eigh(H)
            eigGrid[indexONE, indexTWO] = [thetas[indexONE], thetas[indexTWO], eigs]
    return eigGrid

# Generate data for three simulations
thetaResolution = NUM_THETA
sim1_data = sim(thetaResolution)
sim2_data = sim(thetaResolution)
sim3_data = sim(thetaResolution)
import matplotlib.gridspec as gridspec

# Create a figure with three panels
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1,1], wspace=0.1)  # tighter space

ax1 = fig.add_subplot(gs[0], projection='3d')
plotEigenvalueMeshHelper(sim1_data, thetaResolution, NUM_STATES, ax1)
# ax1.set_title("Simulation 1")

ax2 = fig.add_subplot(gs[1], projection='3d')
plotEigenvalueMeshHelper(sim2_data, thetaResolution, NUM_STATES, ax2)
# ax2.set_title("Simulation 2")

ax3 = fig.add_subplot(gs[2], projection='3d')
plotEigenvalueMeshHelper(sim3_data, thetaResolution, NUM_STATES, ax3)
# ax3.set_title("Simulation 3")

plt.tight_layout()
plt.savefig("threeEnergySpacings6.pdf",bbox_inches="tight",pad_inches=0.25)
# plt.show()

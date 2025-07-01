Chern Numbers in the Lowest Landau Level Dataset Guide:

This dataset is around ~250GB. The dataset is compressed in segments: the directory for each system-size is compressed as a separate archive, for efficiency. Tar + zstd compression is used: the archives "N=128.tar.zst" must thus be decompressed.

The following command may be used for decompression:

```
tar --use-compress-program=unzstd -xvf N=128.tar.zst
```

and similarly for other system-sizes.

The file-sizes for each system-size archive are as follows:

| System Size $N_\phi$ | File Size (GB) |
|----------------------|----------------|
| 8                    | 18.43 GB       |
| 16v1                 | 16.95 GB       |
| 16v2                 | 16.95 GB       |
| 32                   | 12.96 GB       |
| 64                   | 20.53 GB       |
| 96                   | 19.23 GB       |
| 128                  | 22.15 GB       |
| 192                  | 19.69 GB       |
| 256                  | 21.88 GB       |
| 512                  | 24.99 GB       |
| 1024                 | 21.96 GB       |
| 2048                 | 20.62 GB       |
| **Total**                | **236.34 GB**      |

Within each system-size directory, there are are files for every trial in .npz format (compressed numpy format). Each .npz file represents data for a single trial: its name contains a timestamp when it was saved and a trial number (which is irrelevant).

An example of opening the data from one .npz file is below:

``` 
def load_trial_data(file_path):
    """Load trial data efficiently from a .npz file."""
    data = np.load(file_path)
    return {
        "PotentialMatrix": data["PotentialMatrix"],
        "ChernNumbers": data["ChernNumbers"],
        "SumChernNumbers": data["SumChernNumbers"],
        "eigs00": data["eigs00"],
        "eigs0pi": data["eigs0pi"],
        "eigsPi0": data["eigsPi0"],
        "eigsPipi": data["eigsPipi"]
    }

There are eigenvalues saved in arrays of length $N_\phi$ at four different boundary conditions: $[0,0], [0,\pi], [\pi,0], [\pi,\pi]$. The number of eigenvalues corresponds to the system-size of the parent directory, $N_\phi$. The eigenvalues are in sorted, increasing order. There is also an array of Chern numbers that corresponds directly to the arrays of eigenvalues.

There is also a parameter SumChernNumbers: this should be validated when analyzing data to ensure the Chern numbers in the trial sum to one. Sum trials have $\sum C\ne1$ although this is rare.

Finally, the Random Potential realization is also saved. The potential matrix is indexed and constructed as in the function below: the matrix should have size parameter ```POTENTIAL_MATRIX_SIZE = int(4*np.sqrt(NUM_STATES))```. It is directly compatible for use in the simulation code given at https://github.com/edeleu/iqhe-simulation.


```
def constructPotential(size, mean=0, stdev=1):
    # define a real, periodic gaussian potential over a size x size field where
    # V(x,y)=sum V_{m,n} exp(2pi*i*m*x/L_x) exp(2pi*i*n*y/L_y)

    # we allow V_(0,0) to be at V[size,size] such that we can have coefficients from V_(-size,-size) to V_(size,size). We want to have both negatively and positively indexed coefficients! 

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
    ```
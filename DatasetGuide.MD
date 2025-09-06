Chern Numbers in the Lowest Landau Level Dataset Guide:

RAW DATA

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

* "16v1" and "16v2" are two separate datasets for $N_\phi=16$. These may be used to validate analysis methods and get a rough idea about the precision of analysis metrics. 

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
```

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

Processed Data

The dataset also exists in a pre-processed form to simplify the loading process which becomes cumbersome with a high number of trials and individual .npz files. Each system-size of the dataset has been pre-processed into .H5 format with GZIP compression which is compatible with a wide variety of languages (Python, Mathematica, Matlab, Julia, etc). This format allows for specific parts of the dataset to be efficiently and quickly be transferred to memory for simple processing.

The structure of the file for each system-size is as follows:
```
/filenames            [num_trials]                                                          dtype=string
/PotentialMatrix      [num_trials, POTENTIAL_MATRIX_SIZE, POTENTIAL_MATRIX_SIZE]            dtype=complex128
/ChernNumbers         [num_trials, $N_\phi$]                                                dtype=int64
/SumChernNumbers      [num_trials]                                                          dtype=float64
/eigs00               [num_trials, $N_\phi$]                                                dtype=float64
/eigs0pi              [num_trials, $N_\phi$]                                                dtype=float64
/eigsPi0              [num_trials, $N_\phi$]                                                dtype=float64
/eigsPipi             [num_trials, $N_\phi$]                                                dtype=float64
```

Iterating through the trials of data in Python may look like the following: 

    import h5py
    from tqdm import tqdm

    file = "N=1024_GZIP.h5"
    with h5py.File(file, "r") as f:
        print(list(f.keys()))   #list available datasets

        # Access og filenames
        filenames = f["filenames"][:]
        num_trials = len(filenames)
        print(num_trials) # number of trials total

        #Next, load desired datasets to memory!
        total_eigsPiPi = f["eigsPipi"][:]  # shape (n_trials, n_eigs)
        total_chernNumbers = f["ChernNumbers"][:]  # shape (n_trials, n_eigs)
        total_sumChernNumbers = f["SumChernNumbers"][:]  # shape (n_trials,)
        # total_potentials = f["PotentialMatrix"]  # shape (n_trials, L, L)

        # Iterate through trials
        for i in tqdm(range(num_trials)):
            if total_sumChernNumbers[i] != 1:
                continue

            eigs = total_eigsPiPi[i]
            cherns = total_chernNumbers[i]

            # process eigs, cherns, whatever...

Loading the data in Mathematica may look like...

    (* list datasets *)
    Import["N=1024.h5", "Datasets"]

    (* trial count *)
    filenames = Import["N=1024.h5", {"Datasets", "filenames"}]
    numTrials = Length[filenames]

    (* load eigs for trial 5 *)
    Import["N=1024.h5", {"Datasets", "eigsPipi", 5}]

    (*Iterate through trials*)
    allTrialEigs = Import["N=1024.h5", {"Datasets", "eigsPipi"}]

As potential matrix data takes up significant amounts of storage, there is also a version of the dataset further processed into .CSV files. Each is organized into several columns:
Trial # | eigs00| eigsPi0 | eigs0Pi | eigsPiPi | chern

Thus, if there are 1,024 eigenvalues in a given trial, it will occupy 1,024 rows. There is another even further processed .CSV that only contains energy values where the condition $|E|< 0.25$ is satisfied by at least one of the four boundary conditions.
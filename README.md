# FRET-data-assimilation

Estimate parameters and hidden states of dynamical models from single-cell FRET data using *variational annealing* (Ye et al, *Phys. Rev. E* **92**, 052901, 2015), a technique of nonlinear data assimilation for pinpointing global minima among highly non-convex cost functions.

Effective numerical optimization of cost functions requires their first (and possibly, second) derivatives. In practice, coding these derivatives can be prohibitive for complicated dynamical models. This code utilizes a NumPy-based implementation of [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), which precludes the need to manually code these derivatives.



## Getting Started

### Prerequisites

FRET-data-assimilation itself is standalone. To run the scripts, the following must be installed on your computer: 

1. Python 2 (tested on version 2.7.13)
2. SciPy (tested on version 0.17.0)
3. NumPy (tested on version 1.10.4 )
4. [VarAnneal](https://github.com/paulrozdeba/varanneal), a Python implementation of variational annealing that utilizes automatic differentiation
5. [PYADOLC](https://github.com/b45ch1/pyadolc), a Python implementation of automatic differentiation

To install items 4 and 5, follow the instructions in the VarAnneal repository readme. 



## Usage

### Define the local data directory

Before doing any estimations, you must define the data directory. The directory in which both input and output data will be stored is defined in src/local_methods.py. The existing repository has a src/local_methods_sample.py, which you should copy to src/local_methods.py (this file is in gitignore). Within the def_data_dir() function, define ```data_dir```  as the absolute path to where your i/o data will be stored. 

### Put recorded data the data directory

Recorded FRET data is currently being saved in the Emonet Lab as .mat structures. These recordings should be saved within a 'recordings' subfolder of ```data_dir```. Recordings may exist in further subfolders of this directory. For example, currently, Emonet Lab FRET data is saved by date (yymmdd), device number, and recording session:

```
data_dir/recordings/170913/Device1/FRET1/FRET_data_workspace.mat
data_dir/recordings/170918/Device2/FRET3/FRET_data_workspace.mat
```

### Generate stimulus and measured data files from MATLAB data

Data assimilation utilizes a stimulus file and a measured data file. If fake data is generated (below), then there may also be a ground truth file. Before performing the estimation, any recorded data from the FRET experimental setup that is saved as a MATLAB structure (which contains recorded FRET data and the delivered stimulus) will need to be parsed into the individual stimulus and measurement files. 

Suppose we have a recorded data set with FRET data from 30 cells, saved in:

```
data_dir/recordings/170913/Device1/FRET1/FRET_data_workspace.mat
```

Working from the ```./scripts``` folder,  we can generate a stimulus and measurement file for cell 7 via:

```python
from gen_py_data_from_mat import gen_py_data_from_mat
gen_py_data_from_mat(dir='170913/Device1/FRET1', mat_file='FRET_data_workspace', cell=17)
```

This will generate the following files (and create the subfolders if they do not exist):

```data_dir/meas_data/
data_dir/meas_data/170913_Device1_FRET1_cell_7.meas
data_dir/stim/170913_Device1_FRET1_cell_7.stim
```

The data is 2 tab-delimited columns; the first consists of the time points, the second are the values.

Sometimes you may have many stimuli and measurement files you want to get a quick visualization of (e.g. to see if the data is garbage or usable). To generate quick plots for all .stim and .meas files that exist, run:

```{python} quick_plots_stim_and_meas
$ python quick_plots_stim_and_meas
```

This will generate .png plots for stimuli and measurements in the respective folders.

### Each estimation -- not data set -- is defined by a unique specs file 

To assimilate a FRET data set, you must create an associated "specs" .txt file. The data assimilation procedure for FRET data consists of 3 steps:

1. Record data or generate synthetic (fake) data
2. Generate many estimates of unknowns using data assimilation with a subset of the data
3. Find optimal parameter estimates by comparing predictions from each estimated variable set against remainder of data

The algorithmic specifications for all 3 steps are saved in a unique specs file. These specs files are stored in the ```data_dir/specs``` subfolder, which you should create. 

The specs file contains all the information about which recorded dataset to use *and* the parameters of the assimilation. It may also contain parameters of generated fake data, if not using recorded data. Since a specs file is unique to an assimilation, and NOT a data set, distinct estimations utilizing the same data set (say, using different model equations or parameter bounds) would each have a distinct specs file. 

A typical specs file looks like:

```
# DATA                  VARIABLE NAME           VALUE
data_var                nD                      2
data_var                nT                      767
data_var                dt                      0.5
data_var                nP                      10
data_var                stim_file               170913_Device1_FRET1_cell_7
data_var                meas_file               170913_Device1_FRET1_cell_7
data_var                meas_noise              [1.0]
data_var                L_idxs                  [1]

# ESTIMATION VARS       VARIABLE NAME           VALUE
est_var                 model                   MWC_MM_2_var_shift
est_var                 bounds_set              1d
est_var                 est_beg_T               5
est_var                 est_end_T               150
est_var                 pred_end_T              300
    

# ESTIMATION SPECS      ESTIMATION SPECS        VALUE
est_spec                est_type                VA
```

Hashed lines are ignored. Each line is a separate algorithmic parameter, pertaining to one of a) data assimilation, b) prediction generation, or c) (if applicable), the fake data generation. Each line contains 3 or more strings, separated by tabs or spaces.

The first string indicates the *type* of specification -- that is, is the variable relevant to data generation and data input (data_var), to estimation procedure (est_var), or to the type of estimation itself (est_spec). The second string is the name of the particular variable or algorithmic parameter that is being set, and the third (or further) string is the value of said variable. Be sure that all variables or values are strings *without* spaces. 

Let us go through some possible variables and their values.

#### ```est_spec``` : The type of estimation

Currently, there is only one ```est_spec```, which is the type of estimation, ```est_type```, and this has only one possible value, ```VA``` -- estimation by variational annealing. This will soon accept other methods such as linear kernel estimation, etc.

#### ```data_var``` / ```est_var```: Specifications of the estimation procedure

Currently, there is no functional distinction between ```est_var``` and ```data_var```. The first is meant for specifications of the actual estimation, whereas the second is meant for specifications on the nature of the input data and the model. At this point, one may simply use ```data_var``` for any of these variables, though it may be handy to keep them distinct for bookkeeping. 

There are several variables that are of this type:

| variable   | variable definition                      | values                                   |
| :--------- | ---------------------------------------- | ---------------------------------------- |
| nD         | dimension of model system                | int; typically 2 for FRET data. must match dimension of model system defined in src/models (see below) |
| nT         | number of timepoints for full FRET trace (both for estimation and predictive cross validation) | int; must equal the number of rows in .meas and .stim |
| nP         | number of parameters to be estimated     | int; must equal the dimension of the parameters in the model system defined in src/models (see below) |
| dt         | timestep of data                         | float; must equal the timestep in the first column of both .meas and .stim |
| stim_file  | stimulus filename                        | str; name (without .stim extension) of stimulus file. If not provided, then the name of the specs file is used instead |
| meas_file  | measurement filename                     | str; name (without .meas extension) of measurement file. If not provided, then the name of the specs file is used instead |
| meas_noise | covariance of measurement data           | list; assumed covariance of each measured variable |
| L_idxs     | measured indices                         | list; indices of measured variables, corresponding to indices of state variables defined in src/models (see below) |
| bounds_set | parameter bounds dictionary within model class to use | str; corresponds to one of the keys in the src/models.my_model.bounds dictionary, where my_model is the model class  (see below) |
| model      | the presumed model equations             | str; must be a class in src/models, following the layout as described in src/models.generic_model_class (see below). |
| est_beg_T  | initial estimation time                  | float; time at which to begin using estimated data; must be less than nT*dt |
| est_end_T  | final estimation time                    | float; time at which estimated data ends; must be less than nT*dt but greater than est_beg_T |
| pred_end_T | final prediction                         | float; ending time of prediction window for cross-validation and error estimation; prediction window begins at est_end_T, when the estimation ends. Should be less than nT*dt but greater than est_end_T |

Importantly, as noted above, one can omit the ```meas_file``` and ```stim_file``` variables if the specs file has the same name as the stimuli and measurement files. 

### Defining model classes 

The model class contains all the information on the dynamical model to which the data is assimilated. A generic  model class with associated methods is listed in src/models.generic_model_class, with comments and notes on how to generate your own model class. The model class contains the attribute ```bounds```, a dictionary whose keys are each themselves dictionaries holding the upper and lower bounds for states and parameter estimation, respectively. The method ```df``` defines the ODEs containing the model dynamics. 

You can use the prescribed models, or there are several examples in src/models to guide you in defining your own.

### Carry out a variational annealing estimation

Finally, with our input data (measurement file and stimulus file), along with specs file, we can run a variational annealing estimate of our FRET data. 

Run a full variational annealing estimate using ```scripts/est_VA.py```. This script accepts as command line arguments a) the specs text file name and b) a seed for random number generators. The seed is used to generate a random initial estimate for the state variables **x** at all points throughout the estimation window (.e.g between est_beg_T and est_pred_T), as well as for all the parameters to be estimated. The initial estimate is chosen uniformly within the bounds of the states or parameters.  This seed is important since due to the nonlinearity of the model dynamics, the cost function is not convex; different initializations may return different minima. Typically, one runs many estimates in parallel on a computing cluster. Later, these will be aggregated to find the optimal estimate.

For specs file ```FRET-1-estimation.txt```, we can perform a variational annealing estimate with initial seed 3:

```
$ python est_VA.py FRET-1-estimation 3

-- Input vars and params loaded from 170913Device1FRET1cell7.txt

Stimulus data imported from 170913Device1FRET1cell7.stim.
Measured data imported from 170913Device1FRET1cell7.meas.
Initializing estimate with seed 3
------------------------------
Step 1 of 61
beta = 0, RF = 1.00000000e-06

Taping action evaluation...
Done!
Time = 0.0296578407288 s

Beginning optimization...
Optimization complete!
Time = 0.306573152542 s
Exit flag = 0
Exit message: CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH
Iterations = 1
Obj. function value = [ 0.35386092]

------------------------------
Step 2 of 61
beta = 1, RF = 2.00000000e-06
...
```

If either the specifications file, measurement file, or stimulus file is missing, it will return an error. The data, which is a pickled object, (.pklz) will be saved in ```data_dir/objects```, within a subfolder whose name is the specs file. Also saved in this subfolder are files containing the estimated parameters, estimated state trajectories, and estimation errors (these saved data are generated by methods of the VarAnneal class).

### Working with fake (simulated) data

### Generating predictions

To find an optimal estimate, we must next cross-validate our parameter estimates with another set of recorded data (the "prediction"). The time window used for the prediction is [est_end_T, end_pred_T], which are set in the specs file. 

For the specs file used above, predictions can be generated by running:

```python
from gen_pred_data import gen_pred_data
gen_pred_data(FRET-1-estimation, IC_range=range(10000))
```

The IC_range is the initializations for which estimations were carried out; i.e. this code would be for 10000 parallel estimations with seeds from 0, ..., 9999. If certain runs are missing in this range, they will simply be skipped. The script will produce a predicted trace from the estimated parameter sets corresponding to each initialization -- here, 10000 predicted traces in all. 

The predicted data is saved as a dictionary:
```python
pred_dict = {'errors': pred_errors, 'path': pred_path}
```
which contains the predicted traces and errors. Again, this dictionary will be saved in the subfolder of ```data_dir/objects``` corresponding to the specs file name. 

Finally, we want to find the optimal estimate given these 10000 predictions. 








## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

[**Nirag Kadakia**](http://nirag.ucsd.edu/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

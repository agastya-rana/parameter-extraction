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



### Tests: (forthcoming)



## Usage

### Create src/local_methods.py and define the local data directory

Before doing any estimations, you must define the data directories. The directory in which both input and output data will be stored is defined in src/local_methods.py. The existing repository has a src/local_methods_sample.py, which you should copy to src/local_methods.py (this file is in gitignore). Within the def_data_dir() function, define ```data_dir```  as the absolute path to where your i/o data will be stored. 

### Put recorded data in data_dir/recordings

Recorded FRET data is currently being saved in the Emonet Lab as .mat structures. These recordings should be saved within a 'recordings' subfolder of the main data directory. Recordings may exist in further subfolders of this directory. For example, currently, Emonet Lab FRET data is saved by date (yymmdd), device number, and recording session:

```
data_dir/recordings/170913/Device1/FRET1/FRET_data_workspace.mat
data_dir/recordings/170918/Device2/FRET3/FRET_data_workspace.mat
```

### Generate stimulus and measured data files from MATLAB data

Data assimilation utilizes a stimulus file and a measured data file. If fake data is generated (below), then there may also be a ground truth file. Before performing the estimation, any recorded data from the FRET experimental setup that is saved as a MATLAB structure (which contains recorded FRET data and the delivered stimulus) will need to be parsed for the individual stimulus and measurement files. 

For example, suppose we have a recorded data set with FRET data from 30 cells, saved in:

```
data_dir/recordings/170913/Device1/FRET1/FRET_data_workspace.mat
```

Working from the /scripts folder,  we can generate a stimulus and measurement file for cell 7 via:

```{python}
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

```python quick_plots_stim_and_meas
$ python quick_plots_stim_and_meas
```

This will generate .png plots for stimuli and measurements in the respective folders.

### Each estimation -- not data set -- is defined by a unique specs file 

The data assimilation procedure for FRET data consists of 3 steps:

1. Record data or generate synthetic (fake) data
2. Generate many estimates of unknowns using data assimilation with a subset of the data
3. Find optimal parameter estimates by comparing predictions from each estimated variable set against remainder of data

The algorithmic specifications for this set of 3 steps are saved in unique text "specs" files. These specs files are stored in the ```data_dir/specs``` folder, which you should create. 

The specs file contains all the information about which recorded data *and* the parameters of the assimilation. It may also contain parameters of generated fake data, if not using recorded data. Since a specs file is unique to an assimilation, and NOT a data set, distinct estimations utilizing the same data set (say, using different model equations or parameter bounds) each will have a distinct specs file. 

A typical specs file looks like:

```
# DATA 				   	VARIABLE NAME			VALUE
data_var				nD						2
data_var				nT						767
data_var				dt						0.5
data_var				nP						10
data_var				meas_noise				[1.0]
data_var				L_idxs					[1]

# ESTIMATION VARS	 	VARIABLE NAME			VALUE
est_var					model					MWC_MM_2_var_shift
est_var					bounds_set				1d
est_var					est_beg_T				5
est_var					est_end_T				150
est_var					pred_end_T				300
	

# ESTIMATION SPECS		ESTIMATION SPECS		VALUE
est_spec				est_type				VA
```

Hashed lines are ignored. Each line is a separate algorithmic parameter, pertaining to one of a) data assimilation, b) prediction generation, or c) (if applicable), the fake data generation. Each line contains 3 or more strings, each separated by tabs or spaces.

The first string indicates the type of specification -- relevant to data generation and data input (data_var), to estimation procedure (est_var), or to the type of estimation itself (est_spec). 

The second string is the particular variable or algorithmic parameter that is being set.

The third (or further) string is the value of said variable. 

#### ```est_spec``` : The type of estimation

Currently, there is only one ```est_spec```, which is the type of estimation, ```est_type```, and this has only one possible value, ```VA```, for estimation by variational annealing. This will soon accept other methods such as linear kernel estimation, etc.

#### ```data_var``` / ```est_var```: Specifications of the estimation procedure

Currently, there is no functional distinction between ```est_var``` and ```data_var```. The first is meant for specifications of the actual estimation, whereas the second is meant for specifications on the nature of the input data and the model. Since there is no distinction in practice, one may simply use ```data_var``` for any of these variables, though it may help to keep them distinct for bookkeeping. 

There are several variables that are of this type:

| variable   | variable definition                      | values                                   |
| :--------- | ---------------------------------------- | ---------------------------------------- |
| nD         | dimension of model system                | int; typically 2 for FRET data. must match dimension of model system defined in src/models (see below) |
| nT         | number of timepoints for full FRET trace (both for estimation and predictive cross validation) | int; must equal the number of rows in .meas and .stim |
| dt         | timestep of data                         | float; must equal the timestep in the first column of both .meas and .stim |
| nP         | number of parameters to be estimated     | int; must equal the dimension of the parameters in the model system defined in src/models (see below) |
| meas_noise | covariance of measurement data           | list; assumed covariance of each measured variable |
| L_idxs     | measured indices                         | list; indices of measured variables, corresponding to indices of state variables defined in src/models (see below) |
| model      | the presumed model equations             | str; must be a class in src/models, following the layout as described in src/models.generic_model_class (see below). |
| bounds_set | parameter bounds dictionary within model class to use | str; corresponds to one of the keys in the src/models.my_model.bounds dictionary, where my_model is the model class  (see below) |
| est_beg_T  | initial estimation time                  | float; time at which to begin using estimated data; must be less than nT*dt |
| est_end_T  | final estimation time                    | float; time at which estimated data ends; must be less than nT*dt |
| pred_end_T | final prediction                         | float; ending time of prediction window for cross-validation and error estimation; prediction window begins at est_end_T, when the estimation ends. |

### Test 







## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

import sys
sys.path.append('..')
sys.path.append('../src')
import json
from src.local_methods import def_data_dir
from src.est_VA import var_anneal, est_VA, create_cell
from src.utils import NumpyEncoder
import numpy as np
main_dir = def_data_dir()


## CASE 1: simulated cell; constant measurement noise; no sparsity
meas_noise = 0.05
## any less and there is no slack in initial conditions (which are not exact), which prevents later estimation


## Define the spec name and file name
sp_name = 'simulation_example_%s' % meas_noise
filename = '%s/specs/%s.txt' % (main_dir, sp_name)


## Define the specs dictionaries and create the specs file
## Uses the following stimulus file to simulate cell with model and params below
data_vars = {'stim_file': 'simulation_example', 'meas_noise': meas_noise*np.ones((1,))} ## dimension of meas_noise is (len(L_idxs),)
## Set the model, parameters, and estimation, prediction windows
est_vars = {'model': 'MWC_linear', 'params_set': [6.0, 0.33, -0.05],
            'est_beg_T': 20, 'est_end_T': 240, 'pred_end_T': 320}
specifications = {'data_vars': data_vars, 'est_vars': est_vars}
with open(filename, 'w') as outfile:
    json.dump(specifications, outfile, indent=4, cls=NumpyEncoder)

## Simulate the cell, and then infer the parameters of the cell back.
output = var_anneal(sp_name, plot=True)


## CASE 1: simulated cell; variable measurement noise; no sparsity
meas_noise = np.linspace(0.04, 0.10, 767).reshape((767,1))
## any less and there is no slack in initial conditions (which are not exact), which prevents later estimation

## Define the spec name and file name
sp_name = 'simulation_example_variable'
filename = '%s/specs/%s.txt' % (main_dir, sp_name)


## Define the specs dictionaries and create the specs file
## Uses the following stimulus file to simulate cell with model and params below
data_vars = {'stim_file': 'simulation_example', 'meas_noise': meas_noise} ## dimension of meas_noise is (len(L_idxs),)
## Set the model, parameters, and estimation, prediction windows
est_vars = {'model': 'MWC_linear', 'params_set': [6.0, 0.33, -0.05],
            'est_beg_T': 20, 'est_end_T': 240, 'pred_end_T': 320}
specifications = {'data_vars': data_vars, 'est_vars': est_vars}
with open(filename, 'w') as outfile:
    json.dump(specifications, outfile, indent=4, cls=NumpyEncoder)

## Simulate the cell, and then infer the parameters of the cell back.
output = var_anneal(sp_name, plot=True)


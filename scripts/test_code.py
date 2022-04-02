import sys

sys.path.append('..')

import json
from src.local_methods import def_data_dir
from src.est_VA import create_cell, est_VA
from src.load_data import load_est_data_VA
import numpy as np
main_dir = def_data_dir()

## Proof of concept steps

## Generate stimulus (1 time thing)

## Make specs file
sp_name = 'simulation_example'
filename = '%s/specs/%s.txt' % (main_dir, sp_name)
data_vars = {'stim_file': 'simulation_example', 'meas_noise': [0.00001]}
## Set the model, parameters, and estimation, prediction windows
est_vars = {'model': 'MWC_linear', 'params_set': [20., 3225., 0.5, 2.0, 6.0, 0.33, -0.05], 'est_beg_T': 30, 'est_end_T': 280, 'pred_end_T': 380}
specifications = {'data_vars': data_vars, 'est_vars': est_vars}
with open(filename, 'w') as outfile:
    json.dump(specifications, outfile, indent=4)

## Run est_va on normal beta range (so default params)
cell = create_cell(filename, save_data=False)
cell = est_VA(filename, cell)

## Manually extract covariance file from storage, for each beta, calculate eigenvalues of matrix and plot
est_dict = load_est_data_VA(filename)
cov = est_dict['params_err']
for beta_idx in range(len(cell.beta_array)):
    covariance = cov[beta_idx, :, :]
    w, v = np.linalg.eig(covariance)
    err = np.array([np.sqrt(covariance[i, i]) for i in range(len(covariance))])
    print(w, err)

## To plot covariance of parameters, write plotting function in plot data that does exactly that
## Figure out how to turn covariance matrix to gaussian using laplace approximation
## https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma


"""
Need to first:
- Plot eigenvalues as a function of beta for a few models and a few input datasets
- Covariance of parameters plot using 2-D or 3-D
- Sparse data implement in the va_ode
- Show analytically that differentiation of cost function to show why eigenvalues should increase linearly with R_f
- Rewrite scripts files with mode of operation
- Check ODE vs SDE, add process noise to model
- Try to infer process noise - this would be super cool
- Stim selection - check notes
"""


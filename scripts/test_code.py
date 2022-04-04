import sys

sys.path.append('..')
sys.path.append('../src')
import json
from src.local_methods import def_data_dir
from src.est_VA import create_cell, est_VA, var_anneal, minimize_pred_error
from src.load_data import load_est_data_VA
import numpy as np
import matplotlib.pyplot as plt
from src.plot_data import plot_params
main_dir = def_data_dir()

## Make specs file
sp_name = 'simulation_example'
filename = '%s/specs/%s.txt' % (main_dir, sp_name)
data_vars = {'stim_file': 'decent_stimulus', 'meas_noise': [0.01]}
## Set the model, parameters, and estimation, prediction windows
est_vars = {'model': 'MWC_linear', 'params_set': [20., 3225., 0.5, 2.0, 6.0, 0.33, -0.05], 'est_beg_T': 30, 'est_end_T': 280, 'pred_end_T': 380}
specifications = {'data_vars': data_vars, 'est_vars': est_vars}
with open(filename, 'w') as outfile:
    json.dump(specifications, outfile, indent=4)

## Run est_va on normal beta range (so default params)
cell = create_cell(sp_name, save_data=False)
cell = est_VA(sp_name, cell)

## Manually extract covariance file from storage, for each beta, calculate eigenvalues of matrix and plot
est_dict = load_est_data_VA(sp_name)
cov = est_dict['params_err']
eigenvalues = np.zeros((len(cell.beta_array), 3))
for beta_idx in range(len(cell.beta_array)):
    covariance = cov[beta_idx, :, :]
    w, v = np.linalg.eig(covariance)
    eigenvalues[beta_idx] = w
    err = np.array([np.sqrt(covariance[i, i]) for i in range(len(covariance))])
    print(w, err)
for i in range(3):
    plt.scatter(cell.beta_array[20:], np.log(eigenvalues[:, i][20:]))
plt.savefig('eigenvalues.png')
plt.close()

import matplotlib.pyplot as plt
import os
out_dir = '%s/plots/%s' % (main_dir, sp_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
print(out_dir)
traj_dict = minimize_pred_error(sp_name)
t = traj_dict['errors']
plt.scatter([i for i in range(len(t))], t)
plt.savefig('%s/traj_err.png' % out_dir)
plt.close()

"""
Need to first:
- Sparse data implement in the va_ode
- Show analytically that differentiation of cost function to show why eigenvalues should increase linearly with R_f
- Rewrite scripts files with mode of operation
- Check ODE vs SDE, add process noise to model
- Try to infer process noise - this would be super cool
- Stim selection - check notes
"""


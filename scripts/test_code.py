import sys

sys.path.append('..')
sys.path.append('../src')
import json
from src.local_methods import def_data_dir
from src.est_VA import create_cell, est_VA, var_anneal
from src.load_data import load_est_data_VA
import numpy as np
import matplotlib.pyplot as plt
main_dir = def_data_dir()

## Proof of concept steps


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
plt.savefig('eigenvals.png')
plt.close()

def gauss(mu, sigma):
    n = 1000
    xstd = np.sqrt(sigma[0,0])
    ystd = np.sqrt(sigma[1,1])
    xmean = mu[0]
    ymean = mu[1]

    x = np.linspace(xmean - 3 * xstd, xmean + 3 * xstd, n)
    y = np.linspace(ymean - 3 * ystd, ymean + 3 * ystd, n)

    ## Non-normalized Gaussian pdf
    X = np.vstack((x, y)).T
    mat_multi = np.dot((X - mu[None, ...]).dot(np.linalg.inv(sigma)), (X - mu[None, ...]).T)
    z = np.diag(np.exp(-1 * (mat_multi)))

    xmesh, ymesh = np.meshgrid(x, y)
    return xmesh, ymesh, z

output = var_anneal(sp_name, plot=True, beta_precision=0.5)
cell = output['cell']
params = output['params']
params_err = output['params_err']
print(params, params_err)
fig, axs = plt.subplots(len(params), len(params), figsize=(15,15))
for i in range(len(params)):
    for j in range(len(params)):
        i_name = cell.model.param_names[cell.model.P_idxs[i]]
        j_name = cell.model.param_names[cell.model.P_idxs[j]]
        mu = np.asarray([params[i], params[j]])
        sigma = np.asarray([[params_err[i, i], params_err[i, j]], [params_err[j, i], params_err[j, j]]])
        x, y, z = gauss(mu, sigma)
        axs[i,j].contour(x,y,z)
plt.savefig("Contours.png")
plt.close()

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


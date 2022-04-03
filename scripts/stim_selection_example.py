"""
Example of selecting an optimal stimulus by generating simulated data and reinferring parameters.

Created by Agastya Rana, 11/18/21.
"""

## TODO: need to update this

import sys
sys.path.append('..')
sys.path.append('../src')
import numpy as np
import json
from src.local_methods import def_data_dir
main_dir = def_data_dir()

## Defining model parameters for different cells
no_cells = 5
## Taken from Kamino (2020)
N_m = 2.018 ## Mean of N
N_sig = 0.387 ## Std dev of N
## Below distribution is arbitrary
a_SS_m = 0.33 ## steady state activity
a_SS_sig = 0.10
slope_m = -0.01 ## slope of dm/dt at a_SS
slope_sig = 0.004
N = np.random.lognormal(N_m, N_sig, no_cells)
a_SS = np.random.normal(a_SS_m, a_SS_sig, no_cells)
slope = np.random.normal(slope_m, slope_sig, no_cells)

# Create range of stimulus generation params
tss = np.linspace(1, 20, 5)
l1s = np.linspace(0, 0.100, 5) ## Delta L in mM
avg_errors = np.zeros((len(tss)*len(l1s),3))
j = 0

## Define the template file
data_vars = {'nD': 2, 'nT': 767, 'nP': 7, 'dt': 0.5, 'stim_type': 'block', 'stim_params': [0, 0],
             'meas_noise': [0.01], 'L_idxs': [1]}
est_vars = {'model': 'MWC_linear', 'params_set': [20., 3225., 0.5, 2.0, 6.0, 0.33, -0.01], 'bounds_set': 'default',
            'est_beg_T': 0, 'est_end_T': 380, 'pred_end_T': 380}
est_specs = {'est_type': 'VA'}
specifications = {'data_vars': data_vars, 'est_vars': est_vars, 'est_specs': est_specs}
with open('%s/specs/stim_select_template.txt' % main_dir, 'w') as outfile:
    json.dump(specifications, outfile, indent=4)


## For each stimulus, try reinferring parameters of each cell
for ts, l1 in [(ts,l1) for ts in tss for l1 in l1s]:
    errors = np.zeros((no_cells))
    ## Apply stimulus to each cell and infer params
    for i in range(no_cells):
        ## Create specs file for each input
        sp_name = '%s_%s_%s' % (ts, l1, i)
        filename = '%s/specs/%s.txt' % (main_dir, sp_name)
        with open('%s/specs/stim_select_template.txt' % main_dir, 'r') as temp:
            specs = json.load(temp)
        specs['data_vars']['stim_params'] = [ts, l1]
        specs['est_vars']['params_set'] = [20.,3225.,0.5,2.0, N[i], a_SS[i], slope[i]]
        with open(filename, 'w') as outfile:
            json.dump(specs, outfile, indent=4)
        ## Run simulate_data which returns reinferred parameters
        pset = simulate_data(sp_name, param_infer=True)
        print(pset)
        errors[i] = ((pset[-3]-N[i])/N[i])**2 + ((pset[-2]-a_SS[i])/a_SS[i])**2 + ((pset[-1]-slope[i])/slope[i])**2
    ## Average error over different cells
    ## Save results of t_s, l1, avg_error
    avg_errors[j] = [ts, l1, np.average(errors)]
    j = j + 1

np.save('%s/results/results.npy' % main_dir, avg_errors)
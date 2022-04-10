import sys
sys.path.append('..')
sys.path.append('../src')

import json
from src.local_methods import def_data_dir
from src.est_VA import var_anneal
import numpy as np
main_dir = def_data_dir()

meas_noise = 0.05 ## any less and there is no slack initial conditions (which are not exact), which prevents later estimation
##sp_name = 'simulation_example_%s' % meas_noise
sp_name = 'simulation_example__decent_%s' % meas_noise
filename = '%s/specs/%s.txt' % (main_dir, sp_name)
## Input is stimulus file, measurement noise; measurement will be simulated according to the model chosen
data_vars = {'stim_file': 'simulation_example', 'meas_noise': meas_noise*np.ones((1, 1))}
## data_vars = {'stim_file': 'decent_stimulus', 'meas_noise': [meas_noise]}
## Set the model, parameters, and estimation, prediction windows
est_vars = {'model': 'MWC_linear', 'params_set': [20., 3225., 0.5, 2.0, 6.0, 0.33, -0.05],
##            'est_beg_T': 14, 'est_end_T': 250, 'pred_end_T': 314}
            'est_beg_T': 20, 'est_end_T': 350, 'pred_end_T': 460}
specifications = {'data_vars': data_vars, 'est_vars': est_vars}
with open(filename, 'w') as outfile:
    json.dump(specifications, outfile, indent=4)

## Does everything you need
output = var_anneal(sp_name, plot=True)
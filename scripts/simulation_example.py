import sys
sys.path.append('..')
sys.path.append('../src')

import json
from src.local_methods import def_data_dir
from src.est_VA import var_anneal
main_dir = def_data_dir()

sp_name = 'simulation_example'
filename = '%s/specs/%s.txt' % (main_dir, sp_name)
## Input is stimulus file, measurement noise; measurement will be simulated according to the model chosen
data_vars = {'stim_file': 'simulation_example', 'meas_noise': [0.01]}
## Set the model, parameters, and estimation, prediction windows
est_vars = {'model': 'MWC_linear', 'params_set': [20., 3225., 0.5, 2.0, 6.0, 0.33, -0.05], 'est_beg_T': 30, 'est_end_T': 280, 'pred_end_T': 380}
specifications = {'data_vars': data_vars, 'est_vars': est_vars}
with open(filename, 'w') as outfile:
    json.dump(specifications, outfile, indent=4)

## Does everything you need
output = var_anneal(sp_name, plot=True)


from src.load_data import load_pred_data
import matplotlib.pyplot as plt
import os
out_dir = '%s/plots/%s' % (main_dir, sp_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
print(out_dir)
pred_dict = load_pred_data(sp_name)
t = pred_dict['traj_err']
plt.scatter([i for i in range(len(t))], t)
plt.savefig('%s/pred_plots_err.png' % out_dir)
plt.close()
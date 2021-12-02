import sys
sys.path.append('..')
sys.path.append('../src')

import json
from src.local_methods import def_data_dir
from src.single_cell_FRET import create_cell_from_mat
from src.simulate_data import simulate_data, gen_pred_data
from src.plot_data import plot_raw_data, pred_plot
main_dir = def_data_dir()

dirname = 'trial_data'
fname = 'FRET_data'
cellno = 18
## Saves stimulus and measurement file from cell cellno in the respective folders.
create_cell_from_mat(dirname, fname, cell=cellno)

sp_name = 'sim_cell_%s' % cellno
filename = '%s/specs/%s.txt' % (main_dir, sp_name)
data_vars = {'nT': 767, 'dt': 0.5, 'stim_file': 'trial_data_cell_%s' % cellno,
             'meas_noise': [0.00001]} ## change this to add measurement noise - can't be 0 otherwise annealing won't work
est_vars = {'model': 'MWC_linear', 'params_set': [20., 3225., 0.5, 2.0, 6.0, 0.33, -0.05], 'bounds_set': 'default',
            'est_beg_T': 30, 'est_end_T': 280, 'pred_end_T': 380}
est_specs = {'est_type': 'VA'}
specifications = {'data_vars': data_vars, 'est_vars': est_vars, 'est_specs': est_specs}
with open(filename, 'w') as outfile:
    json.dump(specifications, outfile, indent=4)

params = simulate_data(sp_name, param_infer=True, save_data=True)
print("Inferred params are: ", params)
gen_pred_data(sp_name)
pred_plot(sp_name)
plot_raw_data()



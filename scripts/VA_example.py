import sys
sys.path.append('..')

import json
from src.single_cell_FRET import create_cell_from_mat
from src.local_methods import def_data_dir
from src.est_VA import est_VA

data_dir = def_data_dir()

# Since the MATLAB data file is stored in example_data_dir/recordings/trial_data/'
# we run the following command to save stimulus and measurement files for an arbitrary cell (e.g. cell 2)
dirname = 'trial_data'
fname = 'FRET_data'
cellno = 18
create_cell_from_mat(dirname, fname, cell=cellno)

# Then, we run the variational annealing algorithm by first generating a specs file, which contains
# variables about the data (data_vars), variables about the estimation being done (est_vars), and the type of
# estimation being done (est_specs).
data_vars = {'nD': 2, 'nT': 767, 'nP': 7, 'dt': 0.5, 'stim_type': 'block', 'stim_params': [0.01],
             'meas_noise': [0.01], 'L_idxs': [1]}
est_vars = {'model': 'MWC_linear', 'params_set': [20., 3225., 0.5, 2.0, 6.0, 0.33, -0.01], 'bounds_set': 'default',
            'est_beg_T': 0, 'est_end_T': 500, 'pred_end_T': 500}
est_specs = {'est_type': 'VA'}
specifications = {'data_vars': data_vars, 'est_vars': est_vars, 'est_specs': est_specs}
spec_name = '%s_cell_%s' % (dirname, cellno)
with open('%s/specs/%s.txt' % (data_dir, spec_name), 'w') as outfile:
    json.dump(specifications, outfile, indent=4)

## Now, we can run the annealing algorithm with an arbitrary seed; this returns the set of optimal parameters
## according to the model.
param_set = est_VA(spec_name, init_seed=0)



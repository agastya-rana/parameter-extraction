import sys
sys.path.append('..')
sys.path.append('../src')
import numpy as npy
import json
from src.utils import NumpyEncoder
from src.local_methods import def_data_dir
from src.est_VA import var_anneal
from src.models import *
main_dir = def_data_dir()

## Import data file and save components
data = npy.loadtxt('%s/sparsity_cell.txt' % main_dir, delimiter=',')
Tt_data = data[:, 0]
Tt_stim = npy.arange(Tt_data[0], Tt_data[-1]+0.01, Tt_data[1]-Tt_data[0])
data_idx = npy.searchsorted(Tt_data, Tt_stim, side='right') - 1
stim_dat = data[:, 1]
stim_vals = stim_dat[data_idx]
FRET = data[:, 2]
MNoise = data[:, 3].reshape((len(data), 1))
stim_file = npy.column_stack((Tt_stim.T, stim_vals.T))
meas_file = npy.column_stack((Tt_data.T, FRET.T))
npy.savetxt('%s/stim/sparsity_cell.stim' % main_dir, stim_file, fmt='%.6f')
npy.savetxt('%s/meas_data/sparsity_cell.meas' % main_dir, meas_file, fmt='%.6f')

model = MWC_MM_Swayam()
model_name = model.__class__.__name__

data_name = "sparsity_cell"
sp_name = data_name + "_" + model_name
filename = "%s/specs/%s.txt" % (main_dir, sp_name)


data_vars = {'stim_file': data_name, 'meas_file': data_name, 'meas_noise': MNoise} ## std*np.ones(nT,) not (nT, 1)
est_vars = {'model': model_name, 'est_beg_T': 9, 'est_end_T': 701, 'pred_end_T': 869}
specifications = {'data_vars': data_vars, 'est_vars': est_vars}
with open(filename, 'w') as outfile:
    json.dump(specifications, outfile, indent=4, cls=NumpyEncoder)

out_dict = var_anneal(sp_name, plot=True, beta_precision=0.1)
out = dict()
out['par'] = out_dict['params']
out['par_err'] = out_dict['params_err']
with open("%s/%s.txt" % (main_dir, sp_name), 'w') as outfile:
    json.dump(out, outfile, indent=4, cls=NumpyEncoder)

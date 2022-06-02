import sys
sys.path.append('..')
sys.path.append('../src')
import numpy as np
import json
from src.utils import NumpyEncoder
from src.local_methods import def_data_dir
from src.est_VA import var_anneal
main_dir = def_data_dir()

batch_no = 220517
cell_no = 0
data_name= "QS_%s_%s" % (batch_no, cell_no)
sp_name = "QS_MWC_MM_Swayam_%s_%s" % (batch_no, cell_no)
filename = "%s/specs/%s.txt" % (main_dir, sp_name)
with open('%s/meas_noise/Meas_noise_%s_%s.txt' % (main_dir, batch_no, cell_no),'r') as f:
    std = np.sqrt(float(f.readlines()[0]))

print('Cell No:', cell_no)
print('Measurement Noise:', std)

data_vars = {'stim_file': data_name , 'meas_file': data_name, 'meas_noise': std*np.ones((1,))}
est_vars = {'model': 'MWC_MM_Swayam', 'est_beg_T': 0, 'est_end_T': 100, 'pred_end_T': 197}
specifications = {'data_vars': data_vars, 'est_vars': est_vars}
with open(filename, 'w') as outfile:
    json.dump(specifications, outfile, indent=4, cls=NumpyEncoder)

out_dict = var_anneal(sp_name, plot=True, beta_precision=1)
params = out_dict['params']
print(params)
N = params[0]
Nerr = out_dict['params_err'][0, 0]

data_name = "BS_%s_%s" % (batch_no, cell_no)
sp_name = "BS_MWC_MM_Swayam_%s_%s" % (batch_no, cell_no)
filename = "%s/specs/%s.txt" % (main_dir, sp_name)

data_vars = {'stim_file': data_name , 'meas_file': data_name, 'meas_noise': std*np.ones((1,))}
est_vars = {'model': 'MWC_MM_Swayam', 'est_beg_T': 0, 'est_end_T': 600, 'pred_end_T': 900,
            'param_bounds': [[N-Nerr, N+Nerr], [0.001, 0.1], [0.001, 0.1]]}
specifications = {'data_vars': data_vars, 'est_vars': est_vars}
with open(filename, 'w') as outfile:
    json.dump(specifications, outfile, indent=4, cls=NumpyEncoder)
out_dict = var_anneal(sp_name, plot=True, beta_precision=0.1)
params = out_dict['params']
print("param", params)
print("err", out_dict['params_err'])
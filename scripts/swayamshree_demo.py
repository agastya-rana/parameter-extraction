import sys
sys.path.append('..')
sys.path.append('../src')
import numpy as np
import json
from src.utils import NumpyEncoder
from src.local_methods import def_data_dir
from src.est_VA import var_anneal
from src.models import *
main_dir = def_data_dir()

batch_no = 220517
cell_no = 0
model = MWC_MM_Swayam()
stim_prot = "QS"

def run_varanneal_swayam_data(stim_prot, batch_no, cell_no, model):

    model_name = model.__class__.__name__
    data_name= "%s_%s_%s" % (stim_prot, batch_no, cell_no)
    sp_name = "%s_%s_%s_%s" % (stim_prot, model_name, batch_no, cell_no)
    filename = "%s/specs/%s.txt" % (main_dir, sp_name)
    mnoise = np.loadtxt('%s/meas_noise/%s_%s_%s.mnoise' % (main_dir, stim_prot, batch_no, cell_no))[:, 1]
    mnoise = mnoise.reshape((len(mnoise),1))

    data_vars = {'stim_file': data_name, 'meas_file': data_name, 'meas_noise': mnoise} ## std*np.ones(nT,) not (nT, 1)
    est_vars = {'model': model_name, 'est_beg_T': 0, 'est_end_T': 155, 'pred_end_T': 197}
    specifications = {'data_vars': data_vars, 'est_vars': est_vars}
    with open(filename, 'w') as outfile:
        json.dump(specifications, outfile, indent=4, cls=NumpyEncoder)

    out_dict = var_anneal(sp_name, plot=True, beta_precision=0.1)
    out = dict()
    out['par'] = out_dict['params']
    out['par_err'] = out_dict['params_err']
    with open("%s/%s.txt" % (main_dir, sp_name), 'w') as outfile:
        json.dump(out, outfile, indent=4, cls=NumpyEncoder)

    params = out_dict['params']
    N = params[0]
    try:
        Nerr = np.sqrt(out_dict['params_err'][0, 0])
    except:
        Nerr = 5

    stim_prot = "BS"
    data_name= "%s_%s_%s" % (stim_prot, batch_no, cell_no)
    sp_name = "%s_%s_%s_%s" % (stim_prot, model_name, batch_no, cell_no)
    filename = "%s/specs/%s.txt" % (main_dir, sp_name)
    data_vars = {'stim_file': data_name , 'meas_file': data_name, 'meas_noise': mnoise}
    est_vars = {'model': model_name, 'est_beg_T': 0, 'est_end_T': 750, 'pred_end_T': 900,
                'param_bounds': [[N-2*Nerr, N+2*Nerr]] + model.param_bounds[1:]}
    specifications = {'data_vars': data_vars, 'est_vars': est_vars}
    with open(filename, 'w') as outfile:
        json.dump(specifications, outfile, indent=4, cls=NumpyEncoder)
    out_dict = var_anneal(sp_name, plot=True, beta_precision=0.1)
    out = dict()
    out['par'] = out_dict['params']
    out['par_err'] = out_dict['params_err']
    print(out)
    with open("%s/results/%s.txt" % (main_dir, sp_name), 'w') as outfile:
        json.dump(out, outfile, indent=4, cls=NumpyEncoder)

run_varanneal_swayam_data(stim_prot, batch_no, cell_no, model)
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

def varanneal_pipeline(stim_prot, cell_name, model, param_bounds=None, est_beg=0, est_end=None, pred_end=None):

    assert isinstance(model, CellModel)
    model_name = model.__class__.__name__
    data_name= "%s_%s" % (stim_prot, cell_name)
    sp_name = "%s_%s_%s" % (stim_prot, model_name, cell_name)
    filename = "%s/specs/%s.txt" % (main_dir, sp_name)
    mnoise_file = np.loadtxt('%s/meas_noise/%s_%s.mnoise' % (main_dir, stim_prot, cell_name))
    tt = len(mnoise_file)
    mnoise = mnoise_file[:, 1].reshape((tt, 1))

    if param_bounds is None:
        param_bounds = model.param_bounds
    if est_beg is None:
        est_beg = mnoise_file[0, 0]
    if est_end is None:
        est_end = mnoise_file[int(4*tt/5), 0]
    if pred_end is None:
        pred_end = mnoise_file[-1, 0]

    data_vars = {'stim_file': data_name, 'meas_file': data_name, 'meas_noise': mnoise} ## std*np.ones(nT,) not (nT, 1)
    est_vars = {'model': model_name, 'est_beg_T': est_beg, 'est_end_T': est_end, 'pred_end_T': pred_end}
    specifications = {'data_vars': data_vars, 'est_vars': est_vars, 'param_bounds': param_bounds}
    with open(filename, 'w') as outfile:
        json.dump(specifications, outfile, indent=4, cls=NumpyEncoder)

    out_dict = var_anneal(sp_name, plot=True, beta_precision=0.1)
    out = dict()
    out['par'] = out_dict['params']
    out['par_err'] = out_dict['params_err']
    with open("%s/results/%s.txt" % (main_dir, sp_name), 'w') as outfile:
        json.dump(out, outfile, indent=4, cls=NumpyEncoder)
    return out

batch_no = 220517
cell_no = 0
cell_name = '%s_%s' % (batch_no, cell_no)
model = MWC_MM_Swayam()

## Conduct the inference for QS
res = varanneal_pipeline("QS", cell_name, model, est_end=155, pred_end=197)

N = res['par'][0]
try:
    Nerr = np.sqrt(res['par_err'][0, 0]) ## this will fail if parameter bounds for QS not estimated
except:
    Nerr = 5 ## in which case we use this default value
new_bounds = [[N-2*Nerr, N+2*Nerr]] + model.param_bounds[1:]

## Use the new bounds for BS
final_res = varanneal_pipeline("BS", cell_name, model, param_bounds=new_bounds, est_end=750, pred_end=900)
print(final_res)


"""
Functions for simulating data from inferred and/or prescribed parameter sets.

Created by Nirag Kadakia and Agastya Rana, 11/18/21.
"""


import sys
import numpy as np
from load_data import load_est_data_VA
from save_data import save_pred_data, save_stim, save_true_states, save_meas_data
from est_VA import est_VA
from load_specs import read_specs_file, compile_all_run_vars
from single_cell_FRET import single_cell_FRET
from local_methods import def_data_dir
main_dir = def_data_dir()

def gen_pred_data(specs_name, seed_range=[0]):
    """
    Generate predictions for prediction time windows using estimated parameter sets from variational annealing estimations.


    Args:
        specs_name: name of specs object from which VA data will be loaded
        seed_range: list of seed values that VA was run on - default is just 0 (stored as [0])
    """

    pred_errors = np.empty(len(seed_range))
    pred_errors[:] = np.nan
    est_path = None
    pred_path = None
    pred_params = None
    num_valids = 0

    for seed in seed_range:
        try:
            data_dict = load_est_data_VA(specs_name, seed)
            print('seed=%s' % seed)
        except:
            print('Seed %s files not found; skipping...' % seed)
            continue
        sys.stdout.flush()
        # Grab obj at final beta; some attributes will be overwritten
        scF = data_dict['obj']
        est_params = data_dict['params'][-1, :]

        # To hold all predicted and estimated paths for all ICs
        if pred_path is None:
            pred_path = np.zeros((len(scF.pred_wind_idxs), scF.nD, len(seed_range)))
        if est_path is None:
            est_path = np.zeros((len(scF.est_wind_idxs), scF.nD, len(seed_range)))
        est_path[:, :, seed] = data_dict['paths'][-1, :, 1:] ## data_dict['paths'][-1] has time col?, and then nD cols
        ## Instead of this, could also run gen_true_states from start;

        # Make pred_params array the first time in the loop
        if pred_params is None:
            pred_params = np.zeros((len(est_params), len(seed_range)))

        # Set the prediction stimuli and grab the meas data in the pred window
        scF.Tt = scF.Tt[scF.pred_wind_idxs]
        scF.stim = scF.stim[scF.pred_wind_idxs]
        scF.meas_data = scF.meas_data[scF.pred_wind_idxs]

        # Generate the forward prediction using estimated parameter dictionary
        scF.x0 = est_path[-1, :, seed]
        scF.gen_true_states()

        pred_errors[seed] = np.sum((scF.true_states[:, scF.L_idxs]
                                  - scF.meas_data) ** 2.0)/len(scF.Tt)
        pred_path[:, :, seed] = scF.true_states
        pred_params[:, seed] = est_params
        num_valids += 1

    if num_valids > 0:
        pred_dict = {'errors': pred_errors, 'pred_path': pred_path,
                     'est_path': est_path, 'params': pred_params}
        save_pred_data(pred_dict, specs_name)
    else:
        print("No valid files. Nothing saved")

def simulate_data(spec_name, save_data=False, param_infer=False):
    """
    Generates simulated data given prescribed parameter set and one of stimulus or stimulus generation protocol.
    If save_data, data is saved; if param_infer, then parameters are reinferred and returned by applying VA to data.
    Args:
        spec_name: name of specs file
    Returns:
        Inferred parameter set
    """
    # Load specifications from file; to be passed to single_cell_FRET object
    list_dict = read_specs_file(spec_name)
    vars_to_pass = compile_all_run_vars(list_dict)
    scF = single_cell_FRET(**vars_to_pass)
    scF.set_stim()
    scF.gen_true_states()
    scF.set_meas_data() ## Need to set noise  to 0 in specs file; note when adding noise, MWC_MM is scaled, MWC_linear not
    if save_data:
        if scF.stim_file is None:
            save_stim(scF, spec_name)
        save_meas_data(scF, spec_name, simulated=True)
        save_true_states(scF, spec_name)
    if param_infer:
        param_set = est_VA(spec_name, scF=scF)
        return param_set



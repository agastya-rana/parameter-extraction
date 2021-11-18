"""
Plot stimulus/measurement data and prediction data.

Created by Nirag Kadakia and Agastya Rana, 11/18/21.
"""

import os
import scipy as sp
from single_cell_FRET import single_cell_FRET
from load_specs import read_specs_file, compile_all_run_vars
from load_data import load_pred_data, load_true_file, load_stim_file, load_meas_file
from save_data import save_opt_pred_plots, save_data_plots
import matplotlib.pyplot as plt
from local_methods import def_data_dir
data_dir = def_data_dir()


def pred_plot(spec_name):
    """
    Plots the prediction based on the most accurate parameter estimation run (out of many trials with different random
    seeds).
    Args:
        spec_name: name of the specs file; the same specs file can be used as the inference specs file
    Returns:
        plot of stimulus, measured data, and prediction
    """

    # Load data from specs file
    list_dict = read_specs_file(spec_name)
    vars_to_pass = compile_all_run_vars(list_dict)
    scF = single_cell_FRET(**vars_to_pass)

    # Load the stimulus and measured file into the scF object
    if scF.stim_file is None:
        scF.stim_file = spec_name
    if scF.meas_file is None:
        scF.meas_file = spec_name
    scF.set_stim()
    scF.set_meas_data()

    # Set the estimation and prediction windows used in plotting
    scF.set_est_pred_windows()

    # Load all of the prediction data and estimation object and dicts
    pred_dict = load_pred_data(spec_name)
    opt_seed = sp.nanargmin(pred_dict['errors'])
    opt_pred_path = pred_dict['pred_path'][:, :, opt_seed]
    est_path = pred_dict['est_path'][:, :, opt_seed]
    opt_params = pred_dict['params'][:, opt_seed]
    est_range = scF.est_wind_idxs
    pred_range = scF.pred_wind_idxs
    full_range = sp.arange(scF.est_wind_idxs[0], scF.pred_wind_idxs[-1])
    est_Tt = scF.Tt[est_range]
    pred_Tt = scF.Tt[pred_range]
    full_Tt = scF.Tt[full_range]

    # Load true state values if using simulated data
    try:
        true_states = load_true_file(spec_name)[:, 1:]
    except:
        true_states = None

    num_plots = len(scF.L_idxs) + 1

    # Plot the stimulus
    plt.subplot(num_plots, 1, 1)
    plt.plot(full_Tt, scF.stim[full_range], color='r', lw=2)
    plt.xlim(full_Tt[0], full_Tt[-1])
    plt.ylim(80, 160)

    # Plotting only observed variables
    for idx in scF.L_idxs:
        plt.subplot(num_plots, 1, iD + 1)
        plt.xlim(full_Tt[0], full_Tt[-1])
        ## Plot Measured Data; could also do this in one line
        plt.plot(est_Tt, scF.meas_data[scF.est_wind_idxs, idx], color='g')
        plt.plot(pred_Tt, scF.meas_data[scF.pred_wind_idxs, idx], color='g')

        ## Plot estimation and prediction (basically inferred data)
        plt.plot(est_Tt, est_path[:, iD], color='r', lw=3)
        plt.plot(pred_Tt, opt_pred_path[:, iD], color='r', lw=3)

        ## Plot true states if this uses fake data
        if true_states is not None:
            plt.plot(scF.Tt, true_states[:, iD], color='k')
    plt.show()

    data = {'full_Tt': full_Tt, 'est_Tt': est_Tt, 'pred_Tt': pred_Tt, 'stim': scF.stim[full_range],
            'meas_data': scF.meas_data[full_range], 'est_path': est_path, 'pred_path': opt_pred_path,
            'params': opt_params}

    save_opt_pred_plots(spec_name, data)

def plot_raw_data(spec_names=None):

    if spec_names == None:
        specs = []
        stim_path = '%s/stim' % data_dir
        meas_path = '%s/meas_data' % data_dir
        data_flags = []
        for (dirpath, dirnames, filenames) in os.walk(stim_path):
            for filename in filenames:
                if filename.endswith('.stim'):
                    spec_name = filename[:-5]
                    if os.path.exists("%s/%s" % (meas_path, spec_name) +".meas"):
                        specs.append(spec_name)
    else:
        specs = spec_names

    for spec in specs:
        stim_Tt = load_stim_file(spec)
        meas_Tt = load_meas_file(spec)
        scF = single_cell_FRET()
        scF.Tt = stim_Tt[:, 0]
        scF.stim = stim_Tt[:, 1:]
        scF.meas_data = meas_Tt[:, 1:]
        save_data_plots(scF, spec)


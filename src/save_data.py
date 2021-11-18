"""
Functions for saving i/o data and analysis data.

Created by Nirag Kadakia at 13:40 10-20-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import gzip
from src.local_methods import def_data_dir

DATA_DIR = def_data_dir()


def save_stim(obj, data_flag):
    out_dir = '%s/stim' % DATA_DIR
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_to_save = sp.vstack((obj.Tt, obj.stim)).T
    sp.savetxt('%s/%s.stim' % (out_dir, data_flag), data_to_save,
               fmt='%.6f', delimiter='\t')


def save_true_states(obj, data_flag):
    out_dir = '%s/true_states' % DATA_DIR
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_to_save = sp.vstack((obj.Tt, obj.true_states.T)).T
    sp.savetxt('%s/%s.true' % (out_dir, data_flag), data_to_save,
               fmt='%.6f', delimiter='\t')


def save_meas_data(obj, spec_name, simulated=False):
    out_dir = '%s/meas_data' % DATA_DIR
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_to_save = sp.vstack((obj.Tt, obj.meas_data.T)).T
    if simulated:
        filename = spec_name+"_simulated"
    else:
        filename = spec_name
    sp.savetxt('%s/%s.meas' % (out_dir, filename), data_to_save,
               fmt='%.6f', delimiter='\t')

def save_pred_data(data_dict, data_flag):
    out_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    filename = '%s/preds.pkl' % out_dir
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
    print('Prediction data saved to %s.' % (data_flag))

def save_estimates(scF, annealer, data_flag):
    out_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    param_set = annealer.save_params('%s/params_seed=%s.npy' % (out_dir, scF.init_seed))
    annealer.save_paths('%s/paths_seed=%s.npy' % (out_dir, scF.init_seed))
    annealer.save_action_errors('%s/action_errors_seed=%s.npy'
                                % (out_dir, scF.init_seed))
    obj_file = ('%s/obj_seed=%s.pklz' % (out_dir, scF.init_seed))
    with gzip.open(obj_file, 'wb') as f:
        pickle.dump(scF, f, protocol=2)
    print('\n%s-%s data saved to %s.' % (data_flag, scF.init_seed, out_dir))
    return param_set

def save_data_plots(scF, spec_name, stim_change=False):
    out_dir = '%s/estimates/%s' % (DATA_DIR, spec_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)
    stimax = plt.subplot(gs[0])
    fretax = plt.subplot(gs[1])
    stimax.plot(scF.Tt, scF.stim)
    stimax.set_ylim(0.08, 0.2)
    fretax.plot(scF.Tt, scF.meas_data)
    if stim_change:
        changes = [True if scF.stim[x] != scF.stim[x + 1] else False for x in range(len(scF.stim) - 1)]
        change_vals = [scF.Tt[x] for x in range(len(scF.stim) - 1) if changes[x]]
        for i in range(len(change_vals)):
            stimax.axvline(x=change_vals[i], color='black', linestyle='--', lw=1, alpha=1.0)
            fretax.axvline(x=change_vals[i], color='black', linestyle='--', lw=1, alpha=1.0)
    plt.show()
    plt.savefig('%s/%s.png' % (out_dir, spec_name))
    plt.close()


def save_meas_plots(scF, data_flag, simulated=False):
    fig = plt.figure()
    num_plots = scF.meas_data.shape[1]

    # Plot measured data (and true states if they exist)
    for iL in range(num_plots):
        plt.subplot(num_plots, 1, iL + 1)
        plt.scatter(scF.Tt, scF.meas_data[:, iL], s=2, color='red')
    out_dir = '%s/meas_data' % DATA_DIR
    plt.savefig('%s/%s.png' % (out_dir, data_flag))
    plt.close()

def save_opt_pred_plots(spec_name, data_dict):
    out_dir = '%s/estimates/%s' % (DATA_DIR, spec_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig('%s/pred_plots.png' % out_dir)
    filename = '%s/preds.pkl' % out_dir
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
    plt.close()

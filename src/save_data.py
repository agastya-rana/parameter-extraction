"""
Functions for saving i/o data and analysis data.

Created by Nirag Kadakia at 13:40 10-20-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""
from __future__ import print_function

import scipy as sp
import os
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import gzip
from local_methods import def_data_dir

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


def save_meas_data(obj, data_flag):
    out_dir = '%s/meas_data' % DATA_DIR
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_to_save = sp.vstack((obj.Tt, obj.meas_data.T)).T
    sp.savetxt('%s/%s.meas' % (out_dir, data_flag), data_to_save,
               fmt='%.6f', delimiter='\t')


def save_estimates(scF, annealer, data_flag):
    out_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    param_set = annealer.save_params('%s/params_IC=%s.npy' % (out_dir, scF.init_seed))
    annealer.save_paths('%s/paths_IC=%s.npy' % (out_dir, scF.init_seed))
    annealer.save_action_errors('%s/action_errors_IC=%s.npy'
                                % (out_dir, scF.init_seed))
    obj_file = ('%s/obj_IC=%s.pklz' % (out_dir, scF.init_seed))
    with gzip.open(obj_file, 'wb') as f:
        pickle.dump(scF, f, protocol=2)
    print('\n%s-%s data saved to %s.' % (data_flag, scF.init_seed, out_dir))
    return param_set

def save_stim_plots(scF, data_flag):
    fig = plt.figure()
    plt.plot(scF.Tt, scF.stim)
    out_dir = '%s/stim' % DATA_DIR
    plt.savefig('%s/%s.png' % (out_dir, data_flag))
    plt.close()


def save_meas_plots(scF, data_flag):
    fig = plt.figure()
    num_plots = scF.meas_data.shape[1]

    # Plot measured data (and true states if they exist)
    for iL in range(num_plots):
        plt.subplot(num_plots, 1, iL + 1)
        plt.scatter(scF.Tt, scF.meas_data[:, iL], s=2, color='red')
    out_dir = '%s/meas_data' % DATA_DIR
    plt.savefig('%s/%s.png' % (out_dir, data_flag))
    plt.close()


def save_pred_data(data_dict, data_flag):
    out_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    filename = '%s/preds.pkl' % out_dir
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
    print('Prediction data saved to %s.' % (data_flag))


def save_opt_pred_plots(data_flag):
    out_dir = '%s/estimates/%s' % (DATA_DIR, data_flag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig('%s/pred_plots.png' % out_dir)
    plt.close()


def save_opt_pred_data(data_flag, stim, meas, est, opt_pred, opt_params):
    out_dir = '%s/estimates/%s' % (DATA_DIR, data_flag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sp.savetxt('%s/stim.txt' % out_dir, stim, fmt='%.4f', delimiter='\t')
    sp.savetxt('%s/meas.txt' % out_dir, meas, fmt='%.4f', delimiter='\t')
    sp.savetxt('%s/est.txt' % out_dir, est, fmt='%.4f', delimiter='\t')
    sp.savetxt('%s/pred.txt' % out_dir, opt_pred, fmt='%.4f', delimiter='\t')
    sp.savetxt('%s/params.txt' % out_dir, opt_params, fmt='%s', delimiter='\t')

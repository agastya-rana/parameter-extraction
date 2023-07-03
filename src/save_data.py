"""
Functions for saving i/o data and analysis data and plots.

Created by Nirag Kadakia at 13:40 10-20-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import numpy as np
import os
import matplotlib
matplotlib.use('agg')
from src.load_data import load_FRET_recording
import pickle
import gzip
from src.local_methods import def_data_dir

DATA_DIR = def_data_dir()

def save_cell_data(dir, mat_file, cell):
    """Save stimulus and measurement files from FRET recording"""
    data = load_FRET_recording(dir, mat_file, cell)
    spec_name = '%s_cell_%s' % (dir.replace('/', '_'), cell)
    out_dir = '%s/stim' % DATA_DIR
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_to_save = np.vstack((data['Tt'], data['stim'])).T
    np.savetxt('%s/%s.stim' % (out_dir, spec_name), data_to_save, fmt='%.6f', delimiter='\t')

    out_dir = '%s/meas_data' % DATA_DIR
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_to_save = np.vstack((data['Tt'], data['FRET'])).T
    np.savetxt('%s/%s.meas' % (out_dir, spec_name), data_to_save, fmt='%.6f', delimiter='\t')


def save_stim(obj, data_flag):
    out_dir = '%s/stim' % DATA_DIR
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_to_save = np.vstack((obj.Tt, obj.stim)).T
    np.savetxt('%s/%s.stim' % (out_dir, data_flag), data_to_save,
               fmt='%.6f', delimiter='\t')


def save_true_states(obj, spec_name):
    out_dir = '%s/true_states' % DATA_DIR
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_to_save = np.vstack((obj.Tt, obj.true_states.T)).T
    np.savetxt('%s/%s.true' % (out_dir, spec_name), data_to_save,
               fmt='%.6f', delimiter='\t')

def save_meas_data(obj, spec_name):
    out_dir = '%s/meas_data' % DATA_DIR
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_to_save = np.vstack((obj.Tt, obj.meas_data.T)).T
    np.savetxt('%s/%s.meas' % (out_dir, spec_name), data_to_save,
               fmt='%.6f', delimiter='\t')

def save_pred_data(data_dict, data_flag):
    out_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filename = '%s/preds.pkl' % out_dir
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
    print('Prediction data saved to %s.' % (data_flag))

def save_annealing(data_dict, data_flag):
    out_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filename = '%s/annealing_results.pkl' % out_dir
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
    print('Varanneal data saved to %s.' % (data_flag))

def save_estimates(scF, annealer, data_flag):
    out_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    annealer.save_params('%s/params_seed=%s.npy' % (out_dir, scF.init_seed))
    annealer.save_params_err('%s/params_err_seed=%s.npy' % (out_dir, scF.init_seed))
    annealer.save_paths('%s/paths_seed=%s.npy' % (out_dir, scF.init_seed))
    annealer.save_action_errors('%s/action_errors_seed=%s.npy'
                                % (out_dir, scF.init_seed))
    obj_file = ('%s/obj_seed=%s.pklz' % (out_dir, scF.init_seed))
    with gzip.open(obj_file, 'wb') as f:
        pickle.dump(scF, f, protocol=2)
    print('\n%s-%s data saved to %s.' % (data_flag, scF.init_seed, out_dir))
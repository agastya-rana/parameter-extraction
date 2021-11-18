"""
Loading data from files for analysis.

Created by Nirag Kadakia and Agastya Rana, 11/18/21.
"""

import scipy as sp
import scipy.io as sio
import pickle
import gzip
from src.local_methods import def_data_dir
DATA_DIR = def_data_dir()

def load_FRET_recording(dir, mat_file, cell):
    """
    Return data dictionary from FRET recording
    Args:
        dir: subdirectory of '/recordings/' where the MATLAB file is stored
        mat_file: MATLAB file where data is stored
        cell: # of cell that one must extract data about

    Returns:
        data_dict: dictionary of FRET data (stimulus, FRET signal) corresponding to cell inputted
    """
    filename = '%s/recordings/%s/%s.mat' % (DATA_DIR, dir, mat_file)
    mat_f = sio.loadmat(filename)

    Tt = mat_f['FRET_data']['image_time'][0, 0][0]
    stim = mat_f['FRET_data']['input_ts'][0, 0][0]
    FRET_idx = mat_f['FRET_data']['cell_%s' % cell][0, 0]['FRET_index'][0, 0][0]

    data_dict = {"Tt": Tt, "stim": stim, "FRET_idx": FRET_idx}
    return data_dict

def load_stim_file(stim_file):
    filename = '%s/stim/%s.stim' % (DATA_DIR, stim_file)
    stim = sp.loadtxt(filename)
    return stim


def load_meas_file(meas_file):
    filename = '%s/meas_data/%s.meas' % (DATA_DIR, meas_file)
    meas_data = sp.loadtxt(filename)
    return meas_data


def load_true_file(true_file):
    filename = '%s/true_states/%s.true' % (DATA_DIR, true_file)
    true_states = sp.loadtxt(filename)
    return true_states

def load_est_data_VA(data_flag, seed):
    in_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
    with gzip.open('%s/obj_IC=%s.pklz' % (in_dir, seed), 'rb') as f:
        obj = pickle.load(f)
    params = sp.load('%s/params_IC=%s.npy' % (in_dir, seed))
    paths = sp.load('%s/paths_IC=%s.npy' % (in_dir, seed))
    errors = sp.load('%s/action_errors_IC=%s.npy' % (in_dir, seed))

    est_dict = dict()
    est_dict['obj'] = obj
    est_dict['params'] = params
    est_dict['paths'] = paths
    est_dict['errors'] = errors

    return est_dict

def load_pred_data(data_flag):
    out_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
    filename = '%s/preds.pkl' % out_dir
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f)

    return data_dict

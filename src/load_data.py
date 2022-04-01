"""
Loading data from files for analysis.

Created by Nirag Kadakia and Agastya Rana, 11/18/21.
"""

import scipy as sp
import scipy.io as sio
import numpy as np
import pickle
import gzip
from src.local_methods import def_data_dir
DATA_DIR = def_data_dir()

def load_FRET_recording(dir, mat_file, cell, updatedmat=False):
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

    if updatedmat:
        Tt = mat_f['data']['t'][0, 0][0]
        stim = sp.array(mat_f['data']['stim'][0, 0][0]) / 1000. ## convert to micro?
        smooth_fret = mat_f['data']['Emed_flat_mat'][0, 0]
        raw_fret = mat_f['data']['Ecorr_flat_mat'][0, 0]
        ## Sample from FRET pdf
        sample_inds = np.floor(np.random.rand(len(smooth_fret[0])) * 1000).astype(int)
        samples = mat_f['E_sample'][0, cell][0]
        fret_sample = samples[sample_inds, :].diagonal()  ## generates a random sample
        lo = mat_f['data']['E_2p27_flat_mat'][0, 0]
        up = mat_f['data']['E_97p73_flat_mat'][0, 0]

        mfret = smooth_fret[cell, :]  ## Median fret signal
        ufret = fret_sample  ## Un-normalized fret
        ufret_lo = lo[cell]
        ufret_up = up[cell]

        nmfret, zero_line, stretch = normalize_FRET(Tt, stim, mfret, extras=True)
        fret = (ufret - zero_line) * stretch
        fret_lo = (ufret_lo - zero_line) * stretch
        fret_up = (ufret_up - zero_line) * stretch
        meas_noise = (fret_up - fret_lo) / 4.  ## each is 2sigma up or down
        data_dict = {"Tt": Tt, "stim": stim, "Median": nmfret,
                     "FRET": # sampled from pdf
                     fret, "noise": meas_noise}
    else:
        Tt = mat_f['FRET_data']['image_time'][0, 0][0]
        stim = mat_f['FRET_data']['input_ts'][0, 0][0]
        FRET_idx = mat_f['FRET_data']['cell_%s' % cell][0, 0]['FRET_index'][0, 0][0]
        nfret = normalize_FRET(Tt, stim, FRET_idx, extras=False)
        data_dict = {"Tt": Tt, "stim": stim, "FRET": nfret}
    return data_dict

def normalize_FRET(Tt, stim, raw_FRET, extras=False):
    changes = [True if stim[x] != stim[x + 1] else False for x in range(len(stim) - 1)]
    change_ind = [x for x in range(len(stim) - 1) if changes[x]]

    norm1_start = change_ind[0]
    norm1_end = change_ind[1]
    norm2_start = change_ind[-2]
    norm2_end = change_ind[-1]

    min1_a = np.amin(raw_FRET[norm1_start:norm1_end + 1])
    min2_a = np.amin(raw_FRET[norm2_start:norm2_end + 1])
    min1_t = Tt[np.argmin(raw_FRET[norm1_start:norm1_end + 1]) + norm1_start]
    min2_t = Tt[np.argmin(raw_FRET[norm2_start:norm2_end + 1]) + norm2_start]

    slope = (min2_a - min1_a) / (min2_t - min1_t)
    zero_line = slope * (Tt - min1_t) + min1_a  ## Making sure both saturating stimuli have 0 activity
    temp = raw_FRET - zero_line  ## Normalize baseline before stretching
    stretch = 1.0 / np.amax(temp[norm2_end:])  ## Stretch factor to have 0 stimulus activity be 1
    fret = (raw_FRET - zero_line) * stretch
    if extras:
        return fret, zero_line, stretch
    else:
        return fret

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

def load_est_data_VA(data_flag, seed=0):
    in_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
    with gzip.open('%s/obj_seed=%s.pklz' % (in_dir, seed), 'rb') as f:
        obj = pickle.load(f)
    params = sp.load('%s/params_seed=%s.npy' % (in_dir, seed))
    paths = sp.load('%s/paths_seed=%s.npy' % (in_dir, seed))
    errors = sp.load('%s/action_errors_seed=%s.npy' % (in_dir, seed))

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

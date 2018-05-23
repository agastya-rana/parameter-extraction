"""
Functions for loading data for analysis.

Created by Nirag Kadakia at 23:30 08-02-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import os
import scipy.io as sio
import h5py
import cPickle
import gzip
from local_methods import def_data_dir


DATA_DIR = def_data_dir()


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
	
def load_FRET_recording(dir, mat_file='FRET_data_workspace', cell=1):
	
	filename = '%s/recordings/%s/%s.mat' % (DATA_DIR, dir, mat_file)
	mat_f = sio.loadmat(filename)
	
	Tt = mat_f['FRET_data']['image_time'][0,0][0]
	stim = mat_f['FRET_data']['input_ts'][0,0][0]
	FRET_idx = mat_f['FRET_data']['cell_%s' % cell][0,0]['FRET_index'][0, 0][0]
	
	data_dict = dict()
	data_dict['Tt'] = Tt
	data_dict['stim'] = stim
	data_dict['FRET_idx'] = FRET_idx
	
	return data_dict
	
def load_est_data_VA(data_flag, IC):
	
	in_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
	
	with gzip.open('%s/obj_IC=%s.pklz' % (in_dir, IC), 'rb') as f:
		obj = cPickle.load(f)
	params = sp.load('%s/params_IC=%s.npy' % (in_dir, IC))
	paths = sp.load('%s/paths_IC=%s.npy' % (in_dir, IC))
	errors = sp.load('%s/action_errors_IC=%s.npy' % (in_dir, IC))
	
	est_dict = dict()
	est_dict['obj'] = obj
	est_dict['params'] = params
	est_dict['paths'] = paths
	est_dict['errors'] = errors
	
	return est_dict
	
def load_pred_data(data_flag):
	
	out_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	
	filename = '%s/preds.pkl' % out_dir
	with open(filename, 'rb') as f:
		data_dict = cPickle.load(f)
		
	return data_dict
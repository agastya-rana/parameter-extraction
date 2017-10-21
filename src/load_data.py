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
import scipy.io as sio
import h5py
from local_methods import def_data_dir


DATA_DIR = def_data_dir()

def load_preliminary_FRET(data_set=1, cell=1):
	filename = '%s/FRET_recordings/preliminary_data/Device1/FRET%s/' \
				'FRET_data_workspace.mat' % (DATA_DIR, data_set)
	f = sio.loadmat(filename)
	
	signal = f['FRET_data']['input_ts'][0,0][0]
	FRET_idx = f['FRET_data']['cell_%s' % cell][0,0]['FRET_index'][0, 0][0]
	
	data = dict()
	data['signal']=signal
	data['FRET_idx'] = FRET_idx
	
	return data

def load_protocol(type='lorenz', params=[1.0]):
	return None

def load_twin_data(data_flags):

	data_dict = dict()

	data_ID  = data_flags[0]
	data_dt = data_flags[1]
	data_sigma = data_flags[2]

	in_dir = '%s/assimilation/%s' % (DATA_DIR, data_ID)
	data_dict['measurements'] = sp.load('%s/measured_states_dt=%s_sigma=%s.npy' 
										% (in_dir, data_dt, data_sigma))
	data_dict['stimuli'] = sp.load('%s/stimulus_dt=%s_sigma=%s.npy' % DATA_DIR)
	
	return data_dict 

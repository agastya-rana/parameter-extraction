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
import scipy.io as sio
import os
from local_methods import def_data_dir


DATA_DIR = def_data_dir()

def save_twin_data(time, states, measured_vars_and_noise=[[0, 1]]):
	
	nL = len(measured_vars_and_noise)
	nT = len(time)
	noisy_states = sp.zeros((nT, nL))
	noise_levels = []
	
	for meas_idx, vars_and_noise in enumerate(measured_vars_and_noise):
		noisy_states[:, meas_idx] = states[:, vars_and_noise[0]]
		noisy_states[:, meas_idx] += sp.random.normal(0, vars_and_noise[1],  nT)
		noise_levels.append(vars_and_noise[1])
					
	true_data = sp.vstack((time, states.T)).T
	twin_data = sp.vstack((time, noisy_states.T)).T
	
	out_dir = '%s/twin_data' % DATA_DIR
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
		
	if nL == 1:
		sp.save('%s/FRET_twin_data_%s.npy' % (out_dir, noise_levels[0]), twin_data)
	else:
		sp.save('%s/FRET_twin_data_%s.npy' % (out_dir, noise_levels[0]), twin_data)
	sp.save('%s/FRET_true_states.npy' % out_dir, true_data)

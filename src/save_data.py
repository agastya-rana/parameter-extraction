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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = def_data_dir()

def save_VA_twin_data(time, states, stimuli, data_flags,
					measured_vars_and_noise=[[0, 1]]):
	
	data_ID = data_flags[0]
	data_dt = data_flags[1]
	data_sigma = data_flags[2]

	nL = len(measured_vars_and_noise)
	nT = len(time)
	noisy_states = sp.zeros((nT, nL))
	noises = []
	
	for meas_idx, vars_and_noise in enumerate(measured_vars_and_noise):
		noisy_states[:, meas_idx] = states[:, vars_and_noise[0]]
		noisy_states[:, meas_idx] += \
			sp.random.normal(0, vars_and_noise[1],  nT)
		noises.append(vars_and_noise[1])
					
	true_data = sp.vstack((time, states.T)).T
	twin_data = sp.vstack((time, noisy_states.T)).T
	
	out_dir = '%s/assimilation/%s' % (DATA_DIR, data_ID)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	
	sp.save('%s/measured_states_dt=%s_sigma=%s.npy' 
				% (out_dir, data_dt, data_sigma), twin_data)
	sp.save('%s/true_states_dt=%s_sigma=%s.npy' 
			% (out_dir, data_dt, data_sigma), true_data)
	sp.save('%s/stimulus_dt=%s_sigma=%s.npy' 
			% (out_dir, data_dt, data_sigma), stimuli)

def save_estimates(annealer, data_flags):
	
	data_ID = data_flags[0]
	data_dt = data_flags[1]
	data_sigma = data_flags[2]
	init_seed = data_flags[3]

	out_dir = '%s/assimilation/%s' % (DATA_DIR, data_ID)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	annealer.save_params('%s/params_dt=%s_sigma=%s_IC=%s.npy' 
							% (out_dir, data_dt, data_sigma, init_seed))
	annealer.save_paths('%s/paths_dt=%s_sigma=%s_IC=%s.npy'
							% (out_dir, data_dt, data_sigma, init_seed))
	annealer.save_action_errors('%s/action_errors_dt=%s_sigma=%s_IC=%s.npy' 
							% (out_dir, data_dt, data_sigma, init_seed))

def save_est_pred_plot(fig, data_flags):

	data_ID = data_flags[0]
	data_dt = data_flags[1]
	data_sigma = data_flags[2]

	out_dir = '%s/assimilation/%s' % (DATA_DIR, data_ID)
	plt.savefig('%s/est_pred_plot_dt=%s_sigma=%s.png' 
					% (out_dir, data_dt, data_sigma))
	plt.savefig('%s/est_pred_plot_dt=%s_sigma=%s.svg' 
					% (out_dir, data_dt, data_sigma))

def save_est_params_plot(fig, data_flags):

    data_ID = data_flags[0]
    data_dt = data_flags[1]
    data_sigma = data_flags[2]

    out_dir = '%s/assimilation/%s' % (DATA_DIR, data_ID)
    plt.savefig('%s/est_params_plot_dt=%s_sigma=%s.png' 
					% (out_dir, data_dt, data_sigma))
    plt.savefig('%s/est_params_plot_dt=%s_sigma=%s.svg' 
					% (out_dir, data_dt, data_sigma))
	

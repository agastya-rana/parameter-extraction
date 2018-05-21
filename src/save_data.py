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
import matplotlib.pyplot as plt
from local_methods import def_data_dir
import pickle

DATA_DIR = def_data_dir()


def save_stim(obj, data_flag):

	out_dir = '%s/stimuli' % DATA_DIR
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	data_to_save = sp.vstack((obj.Tt, obj.stim)).T
	sp.savetxt('%s/%s.txt' % (out_dir, data_flag), data_to_save, 
				fmt='%.6f', delimiter='\t')
		
def save_true_states(obj, data_flag):

	out_dir = '%s/true_states' % DATA_DIR
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	data_to_save = sp.vstack((obj.Tt, obj.true_states.T)).T
	sp.savetxt('%s/%s.txt' % (out_dir, data_flag), data_to_save, 
				fmt='%.6f', delimiter='\t')

def save_meas_data(obj, data_flag):

	out_dir = '%s/meas_data' % DATA_DIR
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	data_to_save = sp.vstack((obj.Tt, obj.meas_data.T)).T
	sp.savetxt('%s/%s.txt' % (out_dir, data_flag), data_to_save, 
				fmt='%.6f', delimiter='\t')


				
				
				
				
				
				
				
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

def save_VA_pred_plot(fig, data_flags):

	data_ID = data_flags[0]
	data_dt = data_flags[1]
	data_sigma = data_flags[2]

	out_dir = '%s/assimilation/%s' % (DATA_DIR, data_ID)
	plt.savefig('%s/VA_pred_plot_dt=%s_sigma=%s.png' 
					% (out_dir, data_dt, data_sigma))
	plt.savefig('%s/VA_pred_plot_dt=%s_sigma=%s.svg' 
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

def save_all_VA_objs(VA_all_objects_dict, data_flags):

	data_ID = data_flags[0]
	data_dt = data_flags[1]
	data_sigma = data_flags[2]

	out_dir = '%s/assimilation/%s' % (DATA_DIR, data_ID)
	filename = '%s/VA_all_objects_dt=%s_sigma=%s.npy' \
				% (out_dir, data_dt, data_sigma)
	with open(filename, 'w') as outfile:
		pickle.dump(VA_all_objects_dict, outfile)

def save_opt_VA_objs(optimal_data_dict, data_flags):
	
	data_ID = data_flags[0]
	data_dt = data_flags[1]
	data_sigma = data_flags[2]

	out_dir = '%s/assimilation/%s' % (DATA_DIR, data_ID)
	filename = '%s/VA_optimal_objects_dt=%s_sigma=%s.npy' \
				% (out_dir, data_dt, data_sigma)
	with open(filename, 'w') as outfile:
		pickle.dump(optimal_data_dict, outfile)

def save_opt_VA_pred_plot(fig, data_flags):

	data_ID = data_flags[0]
	data_dt = data_flags[1]
	data_sigma = data_flags[2]

	plt.tight_layout()

	out_dir = '%s/assimilation/%s' % (DATA_DIR, data_ID)
	plt.savefig('%s/opt_VA_pred_plot_dt=%s_sigma=%s.png'
				% (out_dir, data_dt, data_sigma))
	plt.savefig('%s/opt_VA_plot_dt=%s_sigma=%s.svg'
                    % (out_dir, data_dt, data_sigma))

def save_estimated_kernels(estimated_kernels, data_flags):

	data_ID = data_flags[0]
	data_dt = data_flags[1]
	data_sigma = data_flags[2]
	kernel_length = data_flags[3]

	out_dir = '%s/assimilation/%s' % (DATA_DIR, data_ID)
	sp.save('%s/kernel_dt=%s_sigma=%s_kernel-length=%s.npy'
			% (out_dir, data_dt, data_sigma, kernel_length), estimated_kernels)

def save_kernel_pred_plot(fig, data_flags):

	data_ID = data_flags[0]
	data_dt = data_flags[1]
	data_sigma = data_flags[2]
	kernel_length = data_flags[3]

	plt.tight_layout()

	out_dir = '%s/assimilation/%s' % (DATA_DIR, data_ID)
	plt.savefig('%s/kernel_pred_plot_dt=%s_sigma=%s_kernel-length=%s.png'
				% (out_dir, data_dt, data_sigma, kernel_length))	
	plt.savefig('%s/kernel_pred_plot_dt=%s_sigma=%s_kernel-length=%s.svg'
                    % (out_dir, data_dt, data_sigma, kernel_length))

def save_opt_kernel_objs(optimal_data_dict, data_flags):
	
	data_ID = data_flags[0]
	data_dt = data_flags[1]
	data_sigma = data_flags[2]
	kernel_length = data_flags[3]

	out_dir = '%s/assimilation/%s' % (DATA_DIR, data_ID)
	filename = '%s/kernel_optimal_objects_dt=%s' \
				'_sigma=%s_kernel-length=%s.npy' \
				% (out_dir, data_dt, data_sigma, kernel_length)
	with open(filename, 'w') as outfile:
		pickle.dump(optimal_data_dict, outfile)
		
def save_opt_kernel_pred_plot(fig, data_flags):

	data_ID = data_flags[0]
	data_dt = data_flags[1]
	data_sigma = data_flags[2]
	kernel_length = data_flags[3]

	plt.tight_layout()
	
	out_dir = '%s/assimilation/%s' % (DATA_DIR, data_ID)
	plt.savefig('%s/opt_kernel_pred_plot_dt=%s_sigma=%s' \
				'_kernel-length=%s.png'
				% (out_dir, data_dt, data_sigma, kernel_length))	
	plt.savefig('%s/opt_kernel_pred_plot_dt=%s_sigma=%s_kernel-length=%s.svg'
                    % (out_dir, data_dt, data_sigma, kernel_length))

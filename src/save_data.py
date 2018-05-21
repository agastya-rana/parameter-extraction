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
import matplotlib.pyplot as plt
import cPickle
import gzip
from local_methods import def_data_dir

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

def save_estimates(scF, annealer, data_flag):
	
	out_dir = '%s/objects/%s' % (DATA_DIR, data_flag)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	
	annealer.save_params('%s/params_IC=%s.npy' % (out_dir, scF.init_seed))
	annealer.save_paths('%s/paths_IC=%s.npy' % (out_dir, scF.init_seed))
	annealer.save_action_errors('%s/action_errors_IC=%s.npy' 
								% (out_dir, scF.init_seed))

	obj_file = ('%s/obj_IC=%s.pklz' % (out_dir, scF.init_seed))
	with gzip.open(obj_file, 'wb') as f:
		cPickle.dump(scF, f, protocol=2)
	print ('\n%s-%s data saved to %s.' % (data_flag, scF.init_seed, out_dir))

def save_stim_and_meas_plots(scF, data_flag, save_plots=True):
	
	fig = plt.figure()
	
	# Plot stimuli
	num_plots = len(scF.L_idxs) + 1
	plt.subplot(num_plots, 1, 1)
	plt.plot(scF.Tt, scF.stim)
	
	# Plot measured data (and true states if they exist)
	for iL_idx, iL in enumerate(scF.L_idxs):
		plt.subplot(num_plots, 1, iL_idx + 2)
		plt.scatter(scF.Tt, scF.meas_data[:, iL_idx], s=2, color='red', 
					label='%s' % scF.model().state_names[iL])
		if scF.true_states is not None:
			plt.plot(scF.Tt, scF.true_states[:, iL])
	plt.legend()
	
	if save_plots == True:
		out_dir = '%s/meas_data' % DATA_DIR
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		plt.savefig('%s/%s.png' % (out_dir, data_flag))
	else:
		plt.show()
	
	
	
	
	
	
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

"""
Prediction plotting from MWC estimates.

Created by Nirag Kadakia at 17:00 10-22-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys, time
sys.path.append('../src')
import scipy as sp
sp.set_printoptions(precision=3)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import get_flags
from single_cell_FRET import single_cell_FRET
from load_data import load_estimated_kernels, load_VA_twin_data
from save_data import save_est_kernel_pred_plot
from params_bounds import *
from models import MWC_Tar


def plot_MWC_kernel_twin_data(data_flags, pred_seed = 10**8):
	"""
	Plot kernels and predicted time series from convolving
	kernels with new stimuli.
	"""	
	
	a = single_cell_FRET()

	data_ID = str(data_flags[0])
	a.dt = float(data_flags[1])
	FRET_noise = float(data_flags[2])
	kernel_length = int(data_flags[3])
	pred_nT = int(data_flags[4])
	pred_density = int(data_flags[5])
	
	estimated_kernels = load_estimated_kernels(data_flags)
	regularization_range = len(estimated_kernels[0, :])

	# Load measured and true trajectories
	data_dict = load_VA_twin_data(data_flags)
	data_EW = data_dict['measurements'][:, 1]
	true_EW = data_dict['true_states'][:, 1:]
	stimuli_EW = data_dict['stimuli']
	est_nT = len(data_EW)
	Tt_EW = sp.arange(0, est_nT*a.dt, a.dt)	
	Tt_PW = sp.arange(Tt_EW[-1], Tt_EW[-1] +  pred_nT*a.dt, a.dt)

	# Generate true prediction and prediction 'data'
	a.nT = pred_nT
	a.set_Tt()
	a.set_step_signal(density=pred_density, seed=pred_seed)
	stimuli_PW = a.signal_vector
	a.model = MWC_Tar
	a.set_true_params(params_dict=params_Tar_1)
	a.x_integrate_init = true_EW[-1, :]
	a.df_integrate()
	true_PW = a.true_states[:, :]
	data_PW = true_PW[:, 1] + sp.random.normal(0, FRET_noise, size=(pred_nT))	

	# Subtract means from stimuli and responses
	mean_subtracted_data_EW = data_EW - sp.average(data_EW)
	mean_subtracted_data_PW = data_PW - sp.average(data_PW)
	mean_subtracted_true_PW = sp.zeros(true_PW.shape)
	for iD in range(a.nD):
		mean_subtracted_true_PW[:, iD] = true_PW[:, iD] - sp.average(true_PW[:, iD])
	mean_subtracted_stimuli_EW = stimuli_EW - sp.average(stimuli_EW)
	mean_subtracted_stimuli_PW = stimuli_PW - sp.average(stimuli_PW)

	# Load estimated states and generate predicted states
	mean_substracted_est_PW = sp.zeros((pred_nT, regularization_range))
	errors_PW = sp.zeros(regularization_range)
	for iR, estimated_kernel in enumerate(estimated_kernels.T):
		a.nT = pred_nT
		a.set_Tt()
		a.signal_vector = stimuli_PW
		mean_substracted_est_PW[:, iR] = a.convolve_stimulus_kernel(estimated_kernel)
		errors_PW[iR] = sp.sum((mean_substracted_est_PW[:, iR] - mean_subtracted_data_PW)**2.0)
	opt_regularization = sp.argmin(errors_PW)
	opt_pred_path = mean_substracted_est_PW[:, opt_regularization]
	opt_pred_kernel = estimated_kernels[:, opt_regularization]

	
	# Plot estimated kernels, predicted response, and stimuli
	fig = plt.figure()
	fig.set_size_inches(10, 8)

	plt.subplot(311)
	for estimated_kernel in estimated_kernels.T:
		plt.plot(sp.arange(kernel_length)*a.dt, estimated_kernel[::-1], 
					color='dodgerblue', lw=0.3)
	plt.plot(sp.arange(kernel_length)*a.dt, opt_pred_kernel[::-1], 
						color='orange', lw=0.5)
	plt.xlim(0, (kernel_length - 1)*a.dt)
	
	plt.subplot(312)
	plt.scatter(Tt_EW, mean_subtracted_data_EW, color='black', zorder=1002, s=0.2)
	plt.scatter(Tt_PW, mean_subtracted_data_PW, color='black', zorder=1002, s=0.2)
	plt.plot(Tt_PW, mean_subtracted_true_PW[:, 1], color='black', zorder=1001, lw=0.5)
	for iR in range(regularization_range):
		plt.plot(Tt_PW, mean_substracted_est_PW[:, iR], color='dodgerblue', lw=0.3)
	plt.plot(Tt_PW, mean_substracted_est_PW[:, opt_regularization], color='orange', zorder=1003, lw=0.5)
	plt.axvline(est_nT*a.dt, ymin=0, ymax=1, color='yellow', lw=0.5)
	plt.xlim(0, a.dt*(pred_nT + est_nT))
	
	plt.subplot(313)
	plt.plot(Tt_EW, mean_subtracted_stimuli_EW, color='dodgerblue')
	plt.plot(Tt_PW, mean_subtracted_stimuli_PW, color='dodgerblue')
	
	save_est_kernel_pred_plot(fig, data_flags)


if __name__ == '__main__':
	data_flags = get_flags()
	plot_MWC_kernel_twin_data(data_flags)	

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


def plot_MWC_kernel_twin_data(data_flags):
	"""
	TODO
	"""	
	
	if len(data_flags) > 4:
		pred_nT = int(data_flags[4])
		pred_density = int(data_flags[5])
	pred_seed = 10**8

	a = single_cell_FRET()

	data_ID = str(data_flags[0])
	a.dt = float(data_flags[1])
	FRET_noise = float(data_flags[2])
	kernel_length = int(data_flags[3])

	# Load estimated kernels
	estimated_kernels = load_estimated_kernels(data_flags)
	regularization_range = len(estimated_kernels[0, :])

	# numpy arrays to hold true/data/estimates; EW = estimation window
	true_EW = None
	true_PW = None
	data_EW = None
	data_PW = None
	est_EW = None
	est_PW = None
	
	# Load measured and true trajectories
	data_dict = load_VA_twin_data(data_flags)
	data_EW = data_dict['measurements']
	true_EW = data_dict['true_states'][:, 1:]
	stimuli_EW = data_dict['stimuli']
	est_nT = len(data_EW[:, 0])
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
	data_PW = true_PW + sp.random.normal(0, FRET_noise, size=(pred_nT, a.nD))	

	# Load estimated states and generate predicted states
	est_EW = sp.zeros((est_nT, regularization_range))
	est_PW = sp.zeros((pred_nT, regularization_range))

	for iK, estimated_kernel in enumerate(estimated_kernels.T):
		a.nT = est_nT
		a.set_Tt()
		a.signal_vector = stimuli_EW
		est_EW[:, iK] = a.convolve_stimulus_kernel(estimated_kernel)

		a.nT = pred_nT
		a.set_Tt()
		a.signal_vector = stimuli_PW
		est_PW[:, iK] = a.convolve_stimulus_kernel(estimated_kernel)

	#opt_IC = sp.argmin(errors_PW)
	#opt_pred_path = est_PW[:, :, opt_IC]

	# Generate prediction and plot traces
	fig = plt.figure()
	fig.set_size_inches(10, 8)

	plt.subplot(211)
	plt.scatter(Tt_EW, data_EW[:, 1], color='black', zorder=1002, s=0.2)
	plt.scatter(Tt_PW, data_PW[:, 1], color='black', zorder=1002, s=0.2)
	plt.plot(Tt_PW, true_PW[:, 1], color='black', zorder=1001, lw=0.5)
	for iR in range(regularization_range):
		plt.plot(Tt_PW, est_PW[:, iR], color='dodgerblue', lw=0.3)
	#plt.plot(Tt_PW, est_PW[:, opt_IC], color='orange', zorder=1003, lw=0.5)
	plt.axvline(est_nT*a.dt, ymin=0, ymax=1, color='yellow', lw=0.5)
	plt.xlim(0, a.dt*(pred_nT + est_nT))
	plt.ylim(-10, 30)

	plt.subplot(212)
	plt.plot(Tt_EW, stimuli_EW)
	plt.plot(Tt_PW, stimuli_PW)
	save_est_kernel_pred_plot(fig, data_flags)


if __name__ == '__main__':
	data_flags = get_flags()
	plot_MWC_kernel_twin_data(data_flags)							

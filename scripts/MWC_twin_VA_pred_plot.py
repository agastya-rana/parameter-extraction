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
from load_data import load_VA_twin_estimates, load_VA_twin_data
from save_data import save_est_VA_pred_plot, save_opt_VA_objs
from params_bounds import *
from models import MWC_Tar


def plot_MWC_VA_prediction_twin_data(data_flags, pred_seed=10**8, beta=50, 
										IC_range=range(10)):
	"""
	Plot prediction of variational annealing time 
	series from forward prediction of the model
	"""	
	
	#TODO Just use this file to generate data; plot from different file

	a = single_cell_FRET()

	data_ID = str(data_flags[0])
	a.dt = float(data_flags[1])
	FRET_noise = float(data_flags[2])
	pred_nT = int(data_flags[3])
	pred_density = int(data_flags[4])

	# Load measured and true trajectories
	data_dict = load_VA_twin_data(data_flags)
	data_EW = data_dict['measurements'][:, 1] 
	true_EW = data_dict['true_states'][:, 1:]
	stimuli_EW = data_dict['stimuli']
	est_nT = len(data_EW)
	Tt_EW = sp.arange(0, est_nT*a.dt, a.dt)	
	Tt_PW = sp.arange(Tt_EW[-1], Tt_EW[-1] + pred_nT*a.dt, a.dt)

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
	
	# Load estimated states and generate predicted states
	est_EW = sp.zeros((est_nT, a.nD, len(IC_range)))
	est_params = sp.zeros((len(a.true_params), len(IC_range)))
	est_PW = sp.zeros((pred_nT, a.nD, len(IC_range)))
	errors_EW = sp.zeros(len(IC_range))
	errors_PW = sp.zeros(len(IC_range))

	for init_seed in IC_range:
		data_dict = load_VA_twin_estimates(data_flags, init_seed)
		est_EW[:, :, init_seed] = data_dict['est_states'][beta, :, 1:]
		est_params[:, init_seed] = data_dict['est_params'][beta, :]
		errors_EW[init_seed] = data_dict['errors'][beta, 1]

		a.nT = pred_nT
		a.set_Tt()
		a.set_step_signal(density=pred_density, seed=pred_seed)
		a.model = MWC_Tar
		a.x_integrate_init = est_EW[-1, :, init_seed]

		# Load estimated params as params for prediction integration
		for iP, Pval in enumerate(est_params[:, init_seed]):
			a.true_params[iP] = Pval

		a.df_integrate()
		est_PW[:, :, init_seed] = a.true_states[:, :]
		errors_PW[init_seed] = sp.sum((est_PW[:, 1, init_seed] - data_PW)**2.0)
		print est_params[:, init_seed]
		print errors_PW[init_seed]
	
	# Find optimal estimate
	opt_IC = sp.argmin(errors_PW)
	opt_pred_path = est_PW[:, :, opt_IC]
	opt_pred_params = est_params[:, opt_IC]

	# Generate prediction and plot traces
	fig = plt.figure()
	fig.set_size_inches(10, 8)

	plt.subplot(311)
	plt.scatter(Tt_EW, data_EW, color='black', zorder=1002, s=0.2)
	plt.scatter(Tt_PW, data_PW, color='black', zorder=1002, s=0.2)
	plt.plot(Tt_PW, true_PW[:, 1], color='black', zorder=1001, lw=0.5)
	for init_seed in IC_range:
		plt.plot(Tt_PW, est_PW[:, 1, init_seed], color='dodgerblue', lw=0.3)
	plt.plot(Tt_PW, opt_pred_path[:, 1], color='orange', zorder=1003, lw=0.5)
	plt.axvline(est_nT*a.dt, ymin=0, ymax=1, color='yellow', lw=0.5)
	plt.xlim(0, a.dt*(pred_nT + est_nT))
	plt.ylim(-10, 30)

	plt.subplot(312)
	plt.plot(Tt_EW, true_EW[:, 0], color='black', zorder=1001)
	plt.plot(Tt_PW, true_PW[:, 0], color='black', zorder=1001)
	for init_seed in IC_range:
		plt.plot(Tt_EW, est_EW[:, 0, init_seed], color='dodgerblue', lw=0.3)
		plt.plot(Tt_PW, est_PW[:, 0, init_seed], color='dodgerblue', lw=0.3)
	plt.plot(Tt_PW, opt_pred_path[:, 0], color='orange', zorder=1002, lw=0.3)
	
	plt.subplot(313)
	plt.plot(Tt_EW, stimuli_EW)
	plt.plot(Tt_PW, stimuli_PW)
	save_est_VA_pred_plot(fig, data_flags)

	# Save optimal objects
	optimal_data_dict = dict()
	opt_objs = ['opt_IC', 'opt_pred_path', 'opt_pred_params', 'Tt_PW',
				'stimuli_PW', 'true_PW', 'data_PW']
	for obj in opt_objs:
		exec('optimal_data_dict["%s"] = %s' % (obj, obj))
	save_opt_VA_objs(optimal_data_dict, data_flags)

	
if __name__ == '__main__':
	data_flags = get_flags()
	plot_MWC_VA_prediction_twin_data(data_flags)							

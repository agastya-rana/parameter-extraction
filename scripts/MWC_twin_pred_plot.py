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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import get_flags
from single_cell_FRET import single_cell_FRET
from load_data import load_VA_twin_estimates, load_VA_twin_data
from save_data import save_est_pred_plot
from params_bounds import *
from models import MWC_Tar


def plot_MWC_prediction_twin_data(data_flags):
	"""
	TODO
	"""	
	
	beta = 35
	pred_nT = 200
	IC_range = range(100)
	pred_seed = 10**8

	a = single_cell_FRET()

	data_ID = str(data_flags[0])
	a.dt = float(data_flags[1])
	FRET_noise = float(data_flags[2])

	# numpy arrays to hold true states and measured data
	true_states = None
	measured_data = None
	true_pred = None

	# numpy arrays to hold estimations and predictions
	est_states = None
	est_params = None
	pred_states = None
	
	# Load measured  data
	measured_data = load_VA_twin_data(data_flags)['measurements']
	est_nT = len(measured_data[:, 0])
	
	true_states = load_VA_twin_data(data_flags)['true_states'][:, 1:]
	a.nT = pred_nT
	a.set_Tt()
	a.set_step_signal(seed=pred_seed)
	a.model = MWC_Tar
	a.set_true_params(params_dict=params_Tar_1)
	a.x_integrate_init = true_states[-1, :]
	a.df_integrate()
	true_pred = a.true_states + sp.random.normal(
							0, FRET_noise, size=(pred_nT, a.nD))
	
	# Load estimated states and generate predicted states
	est_states = sp.zeros((est_nT, a.nD, len(IC_range)))
	est_params = sp.zeros((len(a.true_params), len(IC_range)))
	pred_states = sp.zeros((pred_nT, a.nD, len(IC_range)))
	errors = sp.zeros(len(IC_range))

	for init_seed in IC_range:
		data_dict = load_VA_twin_estimates(data_flags, init_seed)
		est_states[:, :, init_seed] = data_dict['est_states'][beta, :, 1:]
		est_params[:, init_seed] = data_dict['est_params'][beta, :]
		errors[init_seed] = data_dict['errors'][beta, 1]

		a.nT = pred_nT
		a.set_Tt()
		a.set_step_signal(seed=pred_seed)
		a.model = MWC_Tar
		a.x_integrate_init = est_states[-1, :, init_seed]

		# Load estimated params as true params for prediction integration
		for iP, Pval in enumerate(est_params[:, init_seed]):
			a.true_params[iP] = Pval

		a.df_integrate()
		pred_states[:, :, init_seed] = a.true_states

	# Find optimal estimate
	opt_IC = sp.argmin(errors)
	opt_pred_path = pred_states[:, :, opt_IC]
	opt_pred_params = est_params[:, opt_IC]

	# Generate prediction and plot traces
	fig = plt.figure()
	plt.scatter(a.Tt, true_pred[:, 1], color='black')

	for init_seed in IC_range:
		plt.plot(a.Tt, pred_states[:, 1, init_seed], 
					color='dodgerblue', linewidth=0.5)

	save_est_pred_plot(fig, data_flags)

	# Generate parameter estimation plots
	print opt_pred_params

if __name__ == '__main__':
	data_flags = get_flags()
	plot_MWC_prediction_twin_data(data_flags)							

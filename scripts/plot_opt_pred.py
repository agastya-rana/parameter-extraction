"""
Plot optimal predictions for variational annealing estimations.


Created by Nirag Kadakia at 20:00 06-22-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys
sys.path.append('../src')
import scipy as sp
from utils import get_flag
from load_data import load_est_data_VA, load_pred_data, load_true_file
from save_data import save_opt_pred_plots, save_opt_pred_data
import matplotlib.pyplot as plt

def pred_plot(data_flag, beta_to_plot=-1):
	
	# Load all of the prediction data and estimation object and dicts
	pred_dict = load_pred_data(data_flag)
	opt_IC = sp.nanargmin(pred_dict['errors'])
	opt_pred_path = pred_dict['path'][:, :, opt_IC]  
	est_dict = load_est_data_VA(data_flag, opt_IC)

	# Set the object, params, and path based on saved data.
	scF = est_dict['obj']
	est_params = est_dict['params']
	est_path = est_dict['paths'][beta_to_plot, :, 1:]
	errors = est_dict['errors'][:, 1]
	
	full_range = range(scF.est_wind_idxs[0], scF.pred_wind_idxs[-1])
	full_Tt = scF.Tt[full_range]
	est_Tt = scF.Tt[scF.est_wind_idxs]
	pred_Tt = scF.Tt[scF.pred_wind_idxs]
	
	# Load true data if using synthetic data
	true_states = None
	try:
		true_states = load_true_file(data_flag)[:, 1:]
	except:
		pass
	
	num_plots = scF.nD + 1
	
	# Plot the stimulus
	plt.subplot(num_plots, 1, 1)
	plt.plot(full_Tt, scF.stim[full_range], color='r', lw=2)
	plt.xlim(full_Tt[0], full_Tt[-1])
	plt.ylim(80, 160)
	
	# Plot the estimates
	iL_idx = 0
	for iD in range(scF.nD):
		plt.subplot(num_plots, 1, iD + 2)
		plt.xlim(full_Tt[0], full_Tt[-1])
	
		if iD in scF.L_idxs:
			
			# Plot measured data
			plt.plot(est_Tt, scF.meas_data[scF.est_wind_idxs, iL_idx], 
						color='g')
			plt.plot(pred_Tt, scF.meas_data[scF.pred_wind_idxs, iL_idx], 
						color='g')
			
			# Plot estimation and prediction
			plt.plot(est_Tt, est_path[:, iD], color='r', lw=3)
			plt.plot(pred_Tt, opt_pred_path[:, iD], color='r', lw=3)
			
			# Plot true states if this uses fake data
			if true_states is not None:
				plt.plot(scF.Tt, true_states[:, iD], color='k')
		
			iL_idx  += 1
		else:
			plt.plot(est_Tt, est_path[:, iD], color='r', lw=3)
			plt.plot(pred_Tt, opt_pred_path[:, iD], color='r', lw=3)
			if true_states is not None:
				plt.plot(scF.Tt, true_states[:, iD], color='k')
	save_opt_pred_plots(data_flag)
	plt.show()
	
	# Save all the optimal predictions, measurement and stimuli to txt files
	stim_to_save = sp.vstack((full_Tt.T, scF.stim[full_range].T)).T
	meas_to_save = sp.vstack((full_Tt.T, scF.meas_data[full_range].T)).T
	est_to_save = sp.vstack((est_Tt.T, est_path.T)).T
	pred_to_save = sp.vstack((pred_Tt.T, opt_pred_path.T)).T
	params_to_save = sp.vstack((scF.model.param_names, est_params[-1, :])).T
	save_opt_pred_data(data_flag, stim_to_save, meas_to_save, 
						est_to_save, pred_to_save, params_to_save)
	
	
if __name__ == '__main__':
	data_flag = get_flag()
	pred_plot(data_flag)	
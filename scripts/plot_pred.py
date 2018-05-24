"""
Plot optimal predictions for variational annealing estimations.


Created by Nirag Kadakia at 20:00 05-22-2018
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
import matplotlib.pyplot as plt

def pred_plot(data_flag, beta_to_plot=58):
	
	pred_dict = load_pred_data(data_flag)
	opt_IC = sp.nanargmin(pred_dict['errors']) + 9
	opt_path = pred_dict['path'][:, :, opt_IC]
	
	est_dict = load_est_data_VA(data_flag, opt_IC)
	
	scF = est_dict['obj']
	est_params = est_dict['params']
	est_path = est_dict['paths'][beta_to_plot, :, 1:]
	true_states = None
	
	try:
		true_states = load_true_file(data_flag)[:, 1:]
	except:
		pass
	
	plt.subplot(311)
	for iL_idx, iL in enumerate(scF.L_idxs):
		plt.scatter(scF.Tt[scF.pred_wind_idxs], 
			scF.meas_data[scF.pred_wind_idxs, iL_idx], color='g', s=3)
		plt.plot(scF.Tt[scF.pred_wind_idxs], opt_path[:, iL], color='r', lw=3)
		plt.plot(scF.Tt[scF.est_wind_idxs], est_path[:, iL], color='r', lw=3)
		plt.scatter(scF.Tt[scF.est_wind_idxs], 
			scF.meas_data[scF.est_wind_idxs, iL_idx], color='g', s=3)
		if true_states is not None:
			plt.plot(scF.Tt, true_states[:, iL], color='k')
	#plt.plot(scF.Tt[scF.pred_wind_idxs], opt_path[:, 0], color='r', lw=3)
	#plt.plot(scF.Tt, scF.stim)
	plt.xlim(scF.Tt[0], scF.Tt[scF.pred_wind_idxs[-1]])
	plt.subplot(312)
	plt.plot(scF.Tt[scF.pred_wind_idxs], opt_path[:, 0], color='r', lw=3)
	plt.xlim(scF.Tt[0], scF.Tt[scF.pred_wind_idxs[-1]])
	plt.subplot(313)
	plt.plot(scF.Tt[scF.pred_wind_idxs], scF.stim[scF.pred_wind_idxs], color='r', lw=2)
	plt.xlim(scF.Tt[0], scF.Tt[scF.pred_wind_idxs[-1]])
	plt.ylim(80, 160)
	plt.show()
	
	for iP, param_name in enumerate(scF.model.param_names):
		print '%s, %.4f' % (param_name, est_params[beta_to_plot, iP])
	
	for iC in range(1000):
		try:
			est_dict = load_est_data_VA(data_flag, iC)
		except:
			continue
		if iC == opt_IC:
			color='r'
			size=15
		else:
			color='k'
			size=3
		plt.scatter(est_dict['params'][beta_to_plot, 3], est_dict['params']
			[beta_to_plot, 4], color=color, s=size)
	plt.show()
	
if __name__ == '__main__':
	data_flag = get_flag()
	pred_plot(data_flag)	
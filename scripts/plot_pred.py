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

def pred_plot(data_flag, beta_to_plot=59):
	
	
	
	pred_dict = load_pred_data(data_flag)
	opt_IC = sp.nanargmin(pred_dict['errors'][:199])
	opt_path = pred_dict['path'][:, :, opt_IC]  
	print pred_dict['errors'][opt_IC]
	print opt_IC
	
	est_dict = load_est_data_VA(data_flag, opt_IC)
	
	scF = est_dict['obj']
	est_params = est_dict['params']
	est_path = est_dict['paths'][beta_to_plot, :, 1:]
	errors = est_dict['errors'][:, 1]
	print errors
	
	true_states = None
	try:
		true_states = load_true_file(data_flag)[:, 1:]
	except:
		pass
	
	plt.subplot(311)
	for iL_idx, iL in enumerate(scF.L_idxs):
		plt.plot(scF.Tt[scF.pred_wind_idxs], opt_path[:, iL], color='r', lw=3)
		plt.plot(scF.Tt[scF.pred_wind_idxs], 
			scF.meas_data[scF.pred_wind_idxs, iL_idx], color='g')#, s=3)
		plt.plot(scF.Tt[scF.est_wind_idxs], est_path[:, iL], color='r', lw=3)
		plt.plot(scF.Tt[scF.est_wind_idxs], 
			scF.meas_data[scF.est_wind_idxs, iL_idx], color='g')#, s=3)
		if true_states is not None:
			plt.plot(scF.Tt, true_states[:, iL], color='k')
	#plt.plot(scF.Tt[scF.pred_wind_idxs], opt_path[:, 0], color='r', lw=3)
	#plt.plot(scF.Tt, scF.stim)
	plt.xlim(scF.Tt[0], scF.Tt[scF.pred_wind_idxs[-1]])
	plt.subplot(312)
	plt.plot(scF.Tt[0: scF.pred_wind_idxs[-1]], scF.stim[0: scF.pred_wind_idxs[-1]], color='r', lw=2)
	plt.xlim(scF.Tt[0], scF.Tt[scF.pred_wind_idxs[-1]])
	plt.ylim(80, 160)
	plt.subplot(313)
	plt.plot(scF.Tt[scF.pred_wind_idxs], opt_path[:, 0], color='r', lw=3)
	plt.xlim(scF.Tt[0], scF.Tt[scF.pred_wind_idxs[-1]])
	
	scF.params_set = 'est_params'
	scF.model.params[scF.params_set] = est_params[beta_to_plot, :]
	#scF.model.params[scF.params_set][5] = 45
	#scF.model.params[scF.params_set][-2] = 42
	# Set the prediction stimuli and grab the meas data in the pred window
	
	scF.Tt = scF.Tt[scF.est_wind_idxs[0]:scF.pred_wind_idxs[-1]]
	scF.stim = scF.stim[scF.est_wind_idxs[0]:scF.pred_wind_idxs[-1]]
	scF.meas_data = scF.meas_data[scF.est_wind_idxs[0]:scF.pred_wind_idxs[-1]]
	
	# Generate the forward prediction using estimated parameter dictionary
	scF.x0 = est_path[scF.est_wind_idxs[0], :]
	scF.gen_true_states()
	
	plt.subplot(311)
	plt.plot(scF.Tt, scF.true_states[:,1], color='b')
	
	
	plt.show()
	
	#TO SAVE
	"""
	Tt_to_save = scF.Tt[scF.est_wind_idxs[0]: scF.pred_wind_idxs[-1]]
	meas_to_save  = scF.meas_data[scF.est_wind_idxs[0]: scF.pred_wind_idxs[-1], :]
	stim_to_save = scF.stim[scF.est_wind_idxs[0]: scF.pred_wind_idxs[-1]]
	#true_to_save = true_states[scF.est_wind_idxs[0]: scF.pred_wind_idxs[-1], 1]
	stim_to_save = sp.vstack((Tt_to_save.T, stim_to_save.T)).T
	meas_to_save = sp.vstack((Tt_to_save.T, meas_to_save.T)).T
	#true_to_save = sp.vstack((Tt_to_save.T, true_to_save.T)).T
	pred_to_save = opt_path
	est_to_save = est_path
	#est_pred = sp.hstack((est_path.T, opt_path[1:, :].T)).T
	#est_pred_to_save = sp.vstack((Tt_to_save.T, est_pred.T)).T
	pred_to_save = sp.vstack((scF.Tt[scF.pred_wind_idxs].T, opt_path.T)).T
	#plt.plot(true_to_save[:, 0], true_to_save[:, 1:])
	plt.plot(meas_to_save[:, 0], meas_to_save[:, 1:])
	plt.plot(pred_to_save[:, 0], pred_to_save[:, 1:])
	plt.show()
	#sp.savetxt('simulation_true.txt', true_to_save, fmt='%.4f', delimiter='\t')
	sp.savetxt('simulation_stim.txt', stim_to_save, fmt='%.4f', delimiter='\t')
	sp.savetxt('simulation_meas.txt', meas_to_save, fmt='%.4f', delimiter='\t')
	sp.savetxt('simulation_pred.txt', pred_to_save, fmt='%.4f', delimiter='\t')
	"""
	
	
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
		plt.scatter(est_dict['params'][beta_to_plot, 1], est_dict['params']
			[beta_to_plot, 5], color=color, s=size)
	plt.show()
	
if __name__ == '__main__':
	data_flag = get_flag()
	pred_plot(data_flag)	
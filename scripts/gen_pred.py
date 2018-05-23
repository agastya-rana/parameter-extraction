"""
Generate prediction data for variational annealing estimations.


Created by Nirag Kadakia at 18:00 05-21-2018
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
from load_data import load_est_data_VA
from save_data import save_pred_data
	

def gen_pred(data_flag, IC_range=range(1000)):
	
	pred_errors = sp.empty(len(IC_range))
	pred_errors[:] = sp.nan
	pred_path = None
	
	for iC in IC_range:
		
		try:
			data_dict = load_est_data_VA(data_flag, iC)
			print 'iC=%s' % iC
		except:
			print '%s_IC=%s.npy not found; skipping...' % (data_flag, iC)
			continue
		sys.stdout.flush()
		
		# Grab obj at final beta; some attributes will be overwritten
		scF = data_dict['obj']
		est_params = data_dict['params'][-1, :]
		est_path = data_dict['paths'][-1, :, 1:]
		if pred_path is None:
			pred_path = sp.zeros((len(scF.pred_wind_idxs), 
							scF.nD, len(IC_range)))
				
		# Set estimated parameters as a new parameter dictionary
		scF.params_set = 'est_params'
		scF.model.params[scF.params_set] = est_params
		
		# Set the prediction stimuli and grab the meas data in the pred window
		scF.stim = scF.stim[scF.pred_wind_idxs]
		scF.meas_data = scF.meas_data[scF.pred_wind_idxs]
		
		# Generate the forward prediction using estimated parameter dictionary
		scF.x0 = est_path[-1, :]
		scF.gen_true_states()
		
		pred_errors[iC] = sp.sum((scF.true_states[:, scF.L_idxs] 
									- scF.meas_data)**2.0)
		pred_path[:, :, iC] = scF.true_states
		
	pred_dict = {'errors': pred_errors, 'path': pred_path}
	save_pred_data(pred_dict, data_flag)
	
	
if __name__ == '__main__':
	data_flag = get_flag()
	gen_pred(data_flag)	
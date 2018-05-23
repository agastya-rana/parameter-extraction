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
from load_data import load_est_data_VA, load_pred_data
import matplotlib.pyplot as plt

def pred_plot(data_flag):
	
	pred_dict = load_pred_data(data_flag)
	opt_IC = sp.nanargmin(pred_dict['errors'])
	opt_path = pred_dict['path'][:, :, opt_IC]
	
	est_dict = load_est_data_VA(data_flag, opt_IC)
	scF = est_dict['obj']
	est_params = est_dict['params']
	
	for iL_idx, iL in enumerate(scF.L_idxs):
		plt.scatter(scF.Tt[scF.pred_wind_idxs], 
			scF.meas_data[scF.pred_wind_idxs, iL_idx], color='r', s=3)
		plt.plot(scF.Tt[scF.pred_wind_idxs], opt_path[:, iL])
		plt.scatter(scF.Tt[scF.est_wind_idxs], 
			scF.meas_data[scF.est_wind_idxs, iL_idx], color='r', s=3)
	print est_params[-1, :]
	plt.show()
	
if __name__ == '__main__':
	data_flag = get_flag()
	pred_plot(data_flag)	
"""
Prediction plotting from MWC estimates.

Created by Nirag Kadakia at 17:00 10-22-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys
sys.path.append('../src')
import scipy as sp
sp.set_printoptions(precision=3)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import get_flags
from load_data import load_all_VA_objs, load_opt_VA_objs
from save_data import save_VA_pred_plot
from plot_formats import VA_pred_fig


def plot_MWC_VA_pred(data_flags):
	"""
	Plot optimal prediction of variational annealing time 
	series from saved dictionary
	"""	
	
	VA_all_objects_dict = load_all_VA_objs(data_flags)	
	for key in VA_all_objects_dict.keys():
		exec('%s = VA_all_objects_dict["%s"]' % (key, key))
	dt = Tt_PW[1] - Tt_PW[0]
	
	VA_opt_objects_dict = load_opt_VA_objs(data_flags)	
	opt_pred_path = VA_opt_objects_dict['opt_pred_path']
	
	fig = VA_pred_fig(dt, Tt_EW, Tt_PW)
		
	plt.subplot(311)
	plt.plot(Tt_EW, stimuli_EW)
	plt.plot(Tt_PW, stimuli_PW)
	
	plt.subplot(312)
	plt.scatter(Tt_EW, data_EW, color='black', zorder=1002, s=0.2)
	plt.scatter(Tt_PW, data_PW, color='black', zorder=1002, s=0.2)
	plt.plot(Tt_PW, true_PW[:, 1], color='black', zorder=1001, lw=0.5)
	for init_seed in IC_range:
		plt.plot(Tt_PW, est_PW[:, 1, init_seed], color='dodgerblue', lw=0.3)
	plt.plot(Tt_PW, opt_pred_path[:, 1], color='orange', zorder=1003, lw=0.5)
	
	plt.subplot(313)
	plt.plot(Tt_EW, true_EW[:, 0], color='black', zorder=1001)
	plt.plot(Tt_PW, true_PW[:, 0], color='black', zorder=1001)
	for init_seed in IC_range:
		plt.plot(Tt_EW, est_EW[:, 0, init_seed], color='dodgerblue', lw=0.3)
		plt.plot(Tt_PW, est_PW[:, 0, init_seed], color='dodgerblue', lw=0.3)
	plt.plot(Tt_PW, opt_pred_path[:, 0], color='orange', zorder=1002, lw=0.3)
	

	save_VA_pred_plot(fig, data_flags)
	
	
if __name__ == '__main__':
	data_flags = get_flags()
	plot_MWC_VA_pred(data_flags)

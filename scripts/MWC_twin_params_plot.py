"""
Parameters plotting for twin data estimations.

Created by Nirag Kadakia at 14:00 10-24-2017
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
from save_data import save_est_params_plot
from params_bounds import *
from models import MWC_Tar


def plot_MWC_parameter_estimates(data_flags):
	"""
	TODO
	"""	
	
	beta = 60
	IC_range = range(1000)
	params_to_plot = [[7, 8], [5,6]]

	a = single_cell_FRET()

	data_ID = str(data_flags[0])
	a.dt = float(data_flags[1])
	FRET_noise = float(data_flags[2])

	a.model = MWC_Tar
	a.set_true_params(params_dict=params_Tar_1)

	true_params = a.true_params	
	est_params = sp.zeros((len(a.true_params), len(IC_range)))
	
	for init_seed in IC_range:
		data_dict = load_VA_twin_estimates(data_flags, init_seed)
		est_params[:, init_seed] = data_dict['est_params'][beta, :]

	# Generate parameter estimation scatter plot
	fig = plt.figure()
	fig.set_size_inches(10, 8)
	num_plots = len(params_to_plot)
	for iPlot, plot_vars in enumerate(params_to_plot):
		plt.subplot(2, int(num_plots/2), iPlot+1 )
		plt.scatter(est_params[plot_vars[0], :], 
					est_params[plot_vars[1], :], c='blue', s=0.4)
		plt.yscale('log')
		plt.xscale('log')
	
	save_est_params_plot(fig, data_flags)

if __name__ == '__main__':
	data_flags = get_flags()
	plot_MWC_parameter_estimates(data_flags)							

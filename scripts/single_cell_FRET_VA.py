"""
Variational annealing of single cell FRET data. 

Created by Nirag Kadakia at 08:00 10-16-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys, time
sys.path.append('../src')
import scipy as sp
from varanneal import va_ode
from utils import get_flags
from single_cell_FRET import single_cell_FRET
from load_data import load_VA_twin_data
from save_data import save_estimates
from params_bounds import *


def single_cell_FRET_VA(data_flags):
	"""
	Function to run a full annealing cycle on FRET data. 
	Uses data saved from MWC_twin_data.py; run this first.
	command line arguments: data_flag, dt, data noise, and 
	initial condition seed.
	"""	

	data_ID = data_flags[0]
	data_dt = float(data_flags[1])
	data_sigma = float(data_flags[2])
	init_seed = int(data_flags[3])

	# Initialize FRET class 
	scF = single_cell_FRET()
	scF.set_param_bounds(bounds_dict=bounds_Tar_3)
	scF.set_state_bounds(bounds_dict=bounds_Tar_3)
	scF.set_bounds()
	scF.Rm = 1.0/data_sigma**2.0

	# Load twin data from file / match scF params
	data_dict = load_VA_twin_data(data_flags=data_flags)
	measurements = data_dict['measurements'][:, 1:]
	stimuli = data_dict['stimuli'][:]
	scF.Tt = data_dict['measurements'][:, 0]
	scF.nT = len(scF.Tt)
	scF.dt = scF.Tt[1]-scF.Tt[0]
	scF.init_seed = init_seed
	scF.initial_estimate()

	# Initalize annealer class
	annealer = va_ode.Annealer()
	annealer.set_model(scF.df_estimation, scF.nD)
	annealer.set_data(measurements, stim=stimuli,  t=scF.Tt)

	# Estimate
	BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 'maxfun':1000000, 
						'maxiter':1000000}
	tstart = time.time()
	annealer.anneal(scF.x_init, scF.p_init, scF.alpha, scF.beta_array, 
					scF.Rm, scF.Rf0, scF.Lidx, scF.Pidx, dt_model=None, 
					init_to_data=True, bounds=scF.bounds, disc='trapezoid', 
					method='L-BFGS-B', opt_args=BFGS_options, 
					adolcID=init_seed)
	print("\nADOL-C annealing completed in %f s."%(time.time() - tstart))

	save_estimates(annealer, data_flags)


if __name__ == '__main__':
	data_flags = get_flags()
	single_cell_FRET_VA(data_flags)

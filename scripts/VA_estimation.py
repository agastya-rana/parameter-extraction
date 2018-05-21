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
#from varanneal import va_ode
from utils import get_flag
from single_cell_FRET import single_cell_FRET
from load_specs import read_specs_file, compile_all_run_vars
from load_data import load_meas_file, load_stim_file



def single_cell_FRET_VA(data_flag, init_seed):
	
	# Load specifications from file; to be passed to single_cell_FRET object
	list_dict = read_specs_file(data_flag)
	vars_to_pass = compile_all_run_vars(list_dict)
	scF = single_cell_FRET(**vars_to_pass)
	
	# Load stimulus and measurements
	scF.set_stim()
	scF.set_meas_data()
	
	# Initalize estimation 
	scF.init_seed = init_seed
	scF.set_init_est()

	# Initalize annealer class
	annealer = va_ode.Annealer()
	annealer.set_model(scF.df_estimation, scF.nD)
	annealer.set_data(measurements, stim=scF.stim,  t=scF.Tt)

	# Set Rm to 
	Rm = 1.0/scF.meas_noise**2.0
	P_idxs = sp.arange(scF.nP)
	
	# Estimate
	BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 'maxfun':1000000, 
						'maxiter':1000000}
	tstart = time.time()
	annealer.anneal(scF.x_init, scF.p_init, scF.alpha, scF.beta_array, 
					Rm, scF.Rf0, scF.L_idxs, P_idxs, dt_model=None, 
					init_to_data=True, bounds=scF.bounds, disc='euler', 
					method='L-BFGS-B', opt_args=BFGS_options, 
					adolcID=init_seed)
	print("\nADOL-C annealing completed in %f s."%(time.time() - tstart))

	save_estimates(annealer, data_flags)


if __name__ == '__main__':
	data_flag = str(sys.argv[1])
	init_seed = int(sys.argv[2])
	single_cell_FRET_VA(data_flag, init_seed)

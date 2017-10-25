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
sp.set_printoptions(precision=3)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import get_flags
from single_cell_FRET import single_cell_FRET
from load_data import load_VA_twin_estimates, load_VA_twin_data
from save_data import save_opt_VA_objs, save_all_VA_objs
from params_bounds import *
from models import MWC_Tar


def MWC_twin_VA_pred_generate(data_flags, pred_seed=10**8, beta=50, 
										IC_range=range(1000)):
	"""
	Generate predictions and predicted errors of variational annealing 
	time from estimated parameters and final states
	"""	
	
	scF = single_cell_FRET()

	data_ID = str(data_flags[0])
	scF.dt = float(data_flags[1])
	FRET_noise = float(data_flags[2])
	pred_nT = int(data_flags[3])
	pred_density = int(data_flags[4])

	# Load measured and true trajectories
	data_dict = load_VA_twin_data(data_flags)
	data_EW = data_dict['measurements'][:, 1] 
	true_EW = data_dict['true_states'][:, 1:]
	stimuli_EW = data_dict['stimuli']
	est_nT = len(data_EW)
	Tt_EW = sp.arange(0, est_nT*scF.dt, scF.dt)	
	Tt_PW = sp.arange(Tt_EW[-1], Tt_EW[-1] + pred_nT*scF.dt, scF.dt)

	# Generate true prediction and prediction 'data'
	scF.nT = pred_nT
	scF.set_Tt()
	scF.set_step_signal(density=pred_density, seed=pred_seed)
	stimuli_PW = scF.signal_vector
	scF.model = MWC_Tar
	scF.set_true_params(params_dict=params_Tar_1)
	scF.x_integrate_init = true_EW[-1, :]
	scF.df_integrate()
	true_PW = scF.true_states[:, :]
	data_PW = true_PW[:, 1] + sp.random.normal(0, FRET_noise, size=(pred_nT))
	
	# Load estimated states and generate predicted states for each run
	est_EW = sp.zeros((est_nT, scF.nD, len(IC_range)))
	est_params = sp.zeros((len(scF.true_params), len(IC_range)))
	est_PW = sp.zeros((pred_nT, scF.nD, len(IC_range)))
	errors_PW = sp.zeros(len(IC_range))

	for init_seed in IC_range:
		print init_seed, 
		est_data_dict = load_VA_twin_estimates(data_flags, init_seed)
		est_EW[:, :, init_seed] = est_data_dict['est_states'][beta, :, 1:]
		est_params[:, init_seed] = est_data_dict['est_params'][beta, :]
		
		scF.nT = pred_nT
		scF.set_Tt()
		scF.set_step_signal(density=pred_density, seed=pred_seed)
		scF.model = MWC_Tar
		scF.x_integrate_init = est_EW[-1, :, init_seed]
		
		# Load estimated parameters; pass to scF object before integrating
		for iP, Pval in enumerate(est_params[:, init_seed]): 
			scF.true_params[iP] = Pval
		scF.df_integrate()
		est_PW[:, :, init_seed] = scF.true_states[:, :]
		errors_PW[init_seed] = sp.sum((est_PW[:, 1, init_seed] - data_PW)**2.0)
	
	# Save all variational annealing estimation objects
	VA_all_objects_dict = dict()
	VA_all_objs = ['Tt_EW', 'Tt_PW', 'data_EW', 'data_PW', 'est_EW', 'est_PW', 
				'true_EW', 'true_PW', 'stimuli_EW', 'stimuli_PW', 'est_params',
				'errors_PW', 'IC_range']
	for obj in VA_all_objs:
		exec('VA_all_objects_dict["%s"] = %s' % (obj, obj))
	save_all_VA_objs(VA_all_objects_dict, data_flags)
		
	# Save optimal objects
	opt_IC = sp.argmin(errors_PW)
	opt_pred_path = est_PW[:, :, opt_IC]
	opt_pred_params = est_params[:, opt_IC]
	
	optimal_data_dict = dict()
	opt_objs = ['opt_IC', 'opt_pred_path', 'opt_pred_params', 'Tt_PW',
				'stimuli_PW', 'true_PW', 'data_PW']
	for obj in opt_objs:
		exec('optimal_data_dict["%s"] = %s' % (obj, obj))
	save_opt_VA_objs(optimal_data_dict, data_flags)

	
if __name__ == '__main__':
	data_flags = get_flags()
	MWC_twin_VA_pred_generate(data_flags)							

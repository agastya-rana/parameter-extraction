"""
Variational annealing of single cell FRET data. 

Created by Nirag Kadakia and Agastya Rana, 11/18/21.
"""
import time

import scipy as sp
from varanneal import va_ode
from src.single_cell_FRET import single_cell_FRET
from src.load_specs import read_specs_file, compile_all_run_vars
from src.save_data import save_estimates

def est_VA(spec_name, scF=None, init_seed=None):
    if scF == None:
        # Load specifications from file; pass to single_cell_FRET object
        list_dict = read_specs_file(spec_name)
        vars_to_pass = compile_all_run_vars(list_dict)
        scF = single_cell_FRET(**vars_to_pass)

        if scF.stim_file is None:
            scF.stim_file = spec_name
        if scF.meas_file is None:
            scF.meas_file = spec_name

        scF.set_stim()
        scF.set_meas_data()

    if init_seed != None:
        scF.init_seed = init_seed
    # Initalize estimation; set the estimation and prediction windows
    scF.set_init_est()
    scF.set_est_pred_windows()

    # Initalize annealer class
    annealer = va_ode.Annealer()
    annealer.set_model(scF.df_estimation, scF.nD)
    annealer.set_data(scF.meas_data[scF.est_wind_idxs, :], stim=scF.stim[scF.est_wind_idxs], t=scF.Tt[scF.est_wind_idxs])

    # Set Rm as inverse covariance; all parameters measured for now
    Rm = 1.0/sp.asarray(scF.meas_noise)**2.0
    P_idxs = sp.arange(scF.nP)

    # Estimate
    BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 'maxfun':1000000, 'maxiter':1000000}
    tstart = time.time()
    annealer.anneal(scF.x_init[scF.est_wind_idxs], scF.p_init,
                    scF.alpha, scF.beta_array, Rm, scF.Rf0,
                    scF.L_idxs, P_idxs, dt_model=None, init_to_data=True,
                    bounds=scF.bounds, disc='trapezoid',
                    method='L-BFGS-B', opt_args=BFGS_options)
    print("\nVariational annealing completed in {} s.".format(time.time() - tstart))
    param_set = save_estimates(scF, annealer, spec_name)
    return param_set

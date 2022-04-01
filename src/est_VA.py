"""
Variational annealing of single cell FRET data. 

Created by Nirag Kadakia and Agastya Rana, 11/18/21.
"""
import time
import scipy as sp
import numpy as np
import sys
from src.load_data import load_est_data_VA, load_pred_data
from varanneal import va_ode
from src.single_cell_FRET import single_cell_FRET
from src.load_specs import read_specs_file, compile_all_run_vars
from src.save_data import save_pred_data, save_stim, save_true_states, save_meas_data, save_estimates
from src.plot_data import plot_trajectories

def simulate_data(spec_name, save_data=False):
    """
    Generates simulated data given prescribed parameter set and one of stimulus or stimulus generation protocol.
    If save_data, data is saved.
    Args:
        spec_name: name of specs file
    Returns:
        Inferred parameter set
    """
    # Load specifications from file; to be passed to single_cell_FRET object
    list_dict = read_specs_file(spec_name)
    vars_to_pass = compile_all_run_vars(list_dict)
    scF = single_cell_FRET(**vars_to_pass)
    scF.set_stim()
    scF.gen_true_states()
    scF.set_meas_data()
    if save_data:
        if scF.stim_file is None:
            save_stim(scF, spec_name)
        save_meas_data(scF, spec_name, simulated=True)
        save_true_states(scF, spec_name)
    return scF

def est_VA(spec_name, scF=None, init_seed=None, save_data=True, beta_inc=1, beta_mid=31, beta_width=30):
    if scF == None:
        scF = simulate_data(spec_name, save_data=save_data)

    if init_seed != None:
        scF.init_seed = init_seed ## influences x and p init
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
    scF.beta_increment = beta_inc
    scF.beta_array = range(beta_mid - beta_inc*beta_width, beta_mid + beta_inc*beta_width, beta_inc)

    # Estimate
    BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 'maxfun':1000000, 'maxiter':1000000}
    tstart = time.time()
    annealer.anneal(scF.x_init[scF.est_wind_idxs], scF.p_init,
                    scF.alpha, scF.beta_array, Rm, scF.Rf0,
                    scF.L_idxs, P_idxs, dt_model=None, init_to_data=True,
                    bounds=scF.bounds, disc='trapezoid',
                    method='L-BFGS-B', opt_args=BFGS_options)
    print("\nVariational annealing completed in {} s.".format(time.time() - tstart))
    param_set, param_err = save_estimates(scF, annealer, spec_name)
    return scF

def var_anneal(spec_name, seed_range=[0], plot=True):
    """
    Run varanneal on the specs file by iteratively choosing better betas, also run over many seeds.
    Store output (optimal beta, optimal trajectory, predicted trajectory, trajectory error) in dictionary
    """
    b_inc = 1
    b_mid = 40
    b_width = 10
    while b_inc > 0.1:
        for seed in seed_range:
            est_VA(spec_name, init_seed=seed, beta_inc=b_inc, beta_mid=b_mid, beta_width=b_width)
        b_mid = minimize_pred_error(spec_name, seed_range)['beta']
        b_inc = b_inc/4
    scF = est_VA(spec_name, init_seed=seed, beta_inc=1, beta_mid=b_mid, beta_width=1)
    trajectory_data = minimize_pred_error(spec_name, seed_range, store_data=True)
    est_path = trajectory_data['opt_traj']
    pred_path = generate_predictions(spec_name)

    if plot:
        plot_trajectories(spec_name, scF, est_path, pred_path)

def generate_predictions(specs_name):
    """
    Generate predictions for prediction time windows using estimated parameter sets from variational annealing estimations.
    Args:
        specs_name: name of specs object from which VA data will be loaded
        seed_range: list of seed values that VA was run on - default is just 0 (stored as [0])
    """

    data = load_pred_data(specs_name)
    scF = load_est_data_VA(specs_name) ## template scF which has the important variables to be used later
    scF.Tt = scF.Tt[scF.pred_wind_idxs]
    scF.stim = scF.stim[scF.pred_wind_idxs]
    scF.meas_data = scF.meas_data[scF.pred_wind_idxs]
    scF.x0 = data['opt_traj'][-1, :]
    scF.params_set = data['params']
    scF.gen_true_states()
    prediction = scF.true_states
    return prediction

def minimize_pred_error(specs_name, seed_range=[0], store_data=False):
    """
    Args:
        specs_name: spec file used
        seed_range: list of seeds used in the varanneal run
        store_data: flag signalling whether trajectories, trajectory errors, optimal betas should be stored

    Returns: the value of beta that minimizes the error in predicted trajectories
    """
    overall_min = None
    best_seed = seed_range[0]
    num_seeds = len(seed_range)
    opt_beta_idxs = np.zeros(num_seeds)
    traj_arr = None
    valid = False
    for seed_idx, seed in enumerate(seed_range):
        try:
            data_dict = load_est_data_VA(specs_name, seed)
        except:
            continue
        scF = data_dict['obj']

        if traj_arr == None:
            traj_arr = np.zeros((len(scF.beta_array), num_seeds))

        if not len(scF.pred_wind_idxs) > 0:
            print("No prediction window to test annealing run")
            sys.exit(1)
        # Set the prediction stimuli and grab the meas data in the pred window
        scF.Tt = scF.Tt[scF.pred_wind_idxs]
        scF.stim = scF.stim[scF.pred_wind_idxs]
        scF.meas_data = scF.meas_data[scF.pred_wind_idxs]
        # Generate the forward prediction using estimated parameter dictionary
        # Choose optimal beta with minimal trajectory error over predictions
        opt_beta_idx = -1
        min_err = None
        for beta_idx in range(len(scF.beta_array)):
            scF.x0 = data_dict['paths'][beta_idx, -1, :]
            scF.params_set = data_dict['params'][beta_idx, :]
            scF.gen_true_states()
            traj_err = np.sum((scF.true_states[:, scF.L_idxs] - scF.meas_data) ** 2.0) / len(scF.Tt)
            traj_arr[beta_idx, seed_idx] = traj_err
            if min_err == None or traj_err < min_err:
                min_err = traj_err
                opt_beta_idx = beta_idx
        if overall_min == None or min_err < overall_min:
            best_seed = seed
            overall_min = min_err
        opt_beta_idxs[seed_idx] = opt_beta_idx
        valid = True

    opt_beta = scF.beta_array[opt_beta_idx]
    if valid:
        traj_errors = np.amin(traj_arr, axis=1)
        data_dict = load_est_data_VA(specs_name, best_seed)
        opt_traj = data_dict['paths'][opt_beta_idx, :, :]
        params = data_dict['params'][opt_beta_idx, :]
        out_dict = {'errors': traj_errors, 'opt_traj': opt_traj, 'params': params, 'beta': opt_beta}
        if store_data:
            save_pred_data(out_dict, specs_name)
        return out_dict
    else:
        print("No valid input files found. Nothing saved.")
        sys.exit(1)
"""
Variational annealing of single cell FRET data. 

Created by Nirag Kadakia and Agastya Rana, 11/18/21.
"""
import numpy as np
from src.load_data import load_est_data_VA, load_pred_data
from src.varanneal import *
from src.single_cell_FRET import single_cell_FRET
from src.load_specs import read_specs_file
from src.save_data import save_pred_data, save_stim, save_true_states, save_meas_data, save_estimates, save_annealing
from src.plot_data import plot_trajectories, plot_params

def create_cell(spec_name, save_data=False):
    list_dict = read_specs_file(spec_name)
    scF = single_cell_FRET(**list_dict)
    if save_data:
        save_stim(scF, spec_name)
        save_meas_data(scF, save_data)
        if scF.true_states != None:
            save_true_states(scF, save_data)
    return scF

def est_VA(spec_name, scF, init_seed=None, save_data=True, beta_inc=1, beta_mid=31, beta_width=30):

    if init_seed != None:
        scF.init_seed = init_seed ## influences x and p init
    # Initalize estimation
    scF.set_init_est() ## to random set of x values across dt, and random p values across all parameters
    # Initalize annealer class
    annealer = Annealer()
    annealer.set_model(scF.df_estimation, scF.nD)
    annealer.set_data(scF.Tt_data[scF.est_data_wind_idxs], scF.meas_data[scF.est_data_wind_idxs, :],
                       scF.Tt[scF.est_wind_idxs], scF.stim[scF.est_wind_idxs])

    # Set Rm as inverse covariance; all parameters measured for now
    Rm = np.reciprocal(np.square(scF.meas_noise[scF.est_data_wind_idxs, :]))
    scF.beta_increment = beta_inc
    scF.beta_array = np.arange(beta_mid - beta_inc*beta_width, beta_mid + beta_inc*beta_width, beta_inc)

    # Estimate
    BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 'maxfun':1000000, 'maxiter':1000000}
    tstart = time.time()
    annealer.anneal(scF.x_init[scF.est_wind_idxs], scF.p_init,
                    scF.alpha, scF.beta_array, Rm, scF.Rf0,
                    scF.L_idxs, init_to_data=True,
                    bounds=scF.bounds, method='L-BFGS-B', opt_args=BFGS_options)
    print("\nVariational annealing completed in {} s.".format(time.time() - tstart))
    save_estimates(scF, annealer, spec_name)
    return scF

def var_anneal(spec_name, scF=None, seed_range=[0], plot=True, beta_precision=0.1, save_data=True):
    """
    Run varanneal on the specs file by iteratively choosing better betas, also run over many seeds.
    Store output (optimal beta, optimal trajectory, predicted trajectory, trajectory error) in dictionary
    """
    b_inc = 1
    b_mid = 31
    b_width = 15
    first = True
    ## Make simulated cell if cell doesn't already exist
    if scF == None:
        scF = create_cell(spec_name)

    while b_inc > beta_precision:
        if first:
            for seed in seed_range:
                est_VA(spec_name, scF, init_seed=seed, beta_inc=b_inc, beta_mid=31, beta_width=30)
            first = False
        else:
            for seed in seed_range:
                est_VA(spec_name, scF, init_seed=seed, beta_inc=b_inc, beta_mid=b_mid, beta_width=b_width)
        b_mid = minimize_pred_error(spec_name, seed_range)['beta']
        b_inc = b_inc/4
    scF = est_VA(spec_name, scF, init_seed=0, beta_inc=0.00001, beta_mid=b_mid, beta_width=0.000001)
    trajectory_data = minimize_pred_error(spec_name, seed_range, store_data=True)
    est_path = trajectory_data['opt_traj']
    pred_path = generate_predictions(spec_name)

    d = load_est_data_VA(spec_name) ## template scF which has the important variables to be used later
    params = d['params'][-1]
    params_err = d['params_err'][-1, :, :]

    if plot:
        plot_trajectories(spec_name, scF, est_path, pred_path, plot_observed=True)
        pnames = scF.model.param_names
        plot_params(params, params_err, pnames, spec_name)

    out_dict = {'cell': scF, 'traj': trajectory_data, 'pred': pred_path, 'params': params, 'params_err': params_err}
    if save_data:
        save_annealing(out_dict, spec_name)
    return out_dict


def generate_predictions(spec_name):
    """
    Generate predictions for prediction time windows using estimated parameter sets from variational annealing estimations.
    Args:
        spec_name: name of specs object from which VA data will be loaded
        seed_range: list of seed values that VA was run on - default is just 0 (stored as [0])
    """

    data = load_pred_data(spec_name)
    scF = load_est_data_VA(spec_name)['obj'] ## template scF which has the important variables to be used later
    scF.Tt = scF.Tt[scF.pred_wind_idxs]
    scF.stim = scF.stim[scF.pred_wind_idxs]
    scF.x0 = data['opt_traj'][-1, 1:]
    scF.params_set = data['params']
    scF.forward_integrate()
    prediction = np.hstack((np.reshape(scF.Tt, (-1, 1)), scF.true_states))
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
        scF.meas_data = scF.meas_data[scF.pred_data_wind_idxs]
        data_times = scF.Tt_data[scF.pred_data_wind_idxs]
        pred_data_idxs = np.searchsorted(scF.Tt, data_times)
        # Generate the forward prediction using estimated parameter dictionary
        # Choose optimal beta with minimal trajectory error over predictions
        opt_beta_idx = -1
        min_err = None
        for beta_idx in range(len(scF.beta_array)):
            scF.x0 = data_dict['paths'][beta_idx, -1, 1:] ## -1 for last time point, 1 for removing time from path
            scF.params_set = data_dict['params'][beta_idx, :]
            scF.forward_integrate()
            traj_err = np.sum((scF.true_states[pred_data_idxs][:, scF.L_idxs] - scF.meas_data) ** 2.0) / len(scF.Tt)
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
        print(traj_errors)
        data_dict = load_est_data_VA(specs_name, best_seed)
        opt_traj = data_dict['paths'][opt_beta_idx, :, :]
        params = data_dict['params'][opt_beta_idx, :]
        params_err = data_dict['params_err'][opt_beta_idx, :, :]
        out_dict = {'errors': traj_errors, 'opt_traj': opt_traj, 'params': params, 'params_err': params_err,
                    'beta': opt_beta}
        if store_data:
            save_pred_data(out_dict, specs_name)
        return out_dict
    else:
        print("No valid input files found. Nothing saved.")
        sys.exit(1)
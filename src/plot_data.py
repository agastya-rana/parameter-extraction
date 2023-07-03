"""
Plot stimulus/measurement data and prediction data.

Created by Nirag Kadakia and Agastya Rana, 11/18/21.
"""

import os
import scipy as sp
import numpy as np
from src.utils import gauss, circle
from single_cell_FRET import single_cell_FRET
from load_data import load_stim_file, load_meas_file
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
from local_methods import def_data_dir
from src.models import *
import json
import seaborn as sns
from scipy.stats import norm, multivariate_normal
data_dir = def_data_dir()

def plot_trajectories(spec_name, scF, est_path=None, pred_path=None, plot_observed=False):

    ## TODO: make sparsity compatible

    # Load all of the prediction data and estimation object and dicts
    full_range = sp.arange(scF.est_wind_idxs[0], scF.pred_wind_idxs[-1])
    est_Tt = scF.Tt[scF.est_wind_idxs]
    pred_Tt = scF.Tt[scF.pred_wind_idxs]
    full_Tt = scF.Tt[full_range]

    if plot_observed:
        fig, axs = plt.subplots(len(scF.L_idxs) + 1, 1, sharex=True)
    else:
        fig, axs = plt.subplots(scF.nD + 1, 1, sharex=True)
    # Plot the stimulus
    axs[0].plot(full_Tt, scF.stim[full_range], color='r', lw=2)
    axs[0].set_xlim(full_Tt[0], full_Tt[-1])
    axs[0].set_ylim(80, 160)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Stimulus ($\mu$M)")

    plot_no = 1
    Lidx = 0
    for num in range(scF.nD):
        if plot_observed and num not in scF.L_idxs:
            continue
        axs[plot_no].set_ylabel(scF.model.state_names[num])
        if num in scF.L_idxs:
            ## Plot Measured Data
            axs[plot_no].plot(full_Tt, scF.meas_data[full_range, Lidx], color='g', label='Measured')
            Lidx += 1
        ## Plot Inferred Data
        if est_path is not None:
            axs[plot_no].plot(est_Tt, est_path[:, num+1], color='r', lw=1, label='Estimated')
        if pred_path is not None:
            axs[plot_no].plot(pred_Tt, pred_path[:, num+1], color='k', lw=1, label='Predicted')
        plot_no += 1
    plt.legend()

    data = {'full_Tt': full_Tt, 'est_Tt': est_Tt, 'pred_Tt': pred_Tt, 'stim': scF.stim[full_range],
            'meas_data': scF.meas_data[full_range], 'est_path': est_path, 'pred_path': pred_path}
    out_dir = '%s/plots/%s' % (data_dir, spec_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig('%s/trajectory.png' % out_dir)
    filename = '%s/trajectory.pkl' % out_dir
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    plt.close()



def plot_raw_data(spec_names=None):
    if spec_names == None:
        specs = []
        stim_path = '%s/stim' % data_dir
        meas_path = '%s/meas_data' % data_dir
        data_flags = []
        for (dirpath, dirnames, filenames) in os.walk(stim_path):
            for filename in filenames:
                if filename.endswith('.stim'):
                    spec_name = filename[:-5]
                    if os.path.exists("%s/%s" % (meas_path, spec_name) +".meas"):
                        specs.append(spec_name)
    else:
        specs = spec_names

    for spec in specs:
        stim_Tt = load_stim_file(spec)
        meas_Tt = load_meas_file(spec)
        scF = single_cell_FRET()
        scF.Tt = stim_Tt[:, 0]
        scF.stim = stim_Tt[:, 1:]
        scF.meas_data = meas_Tt[:, 1:]
        plot_exp(scF, spec)

def plot_exp(scF, spec_name, stim_change=False):
    out_dir = '%s/plots/%s' % (data_dir, spec_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig = plt.figure(figsize=(10, 10))
    fig, (stimax, fretax) = plt.subplots(2, 1, sharex=True)
    stimax.plot(scF.Tt, scF.stim)
    stimax.set_ylim(80, 200)
    stimax.set_ylabel('Stimulus (uM)')
    fretax.plot(scF.Tt, scF.meas_data)
    fretax.set_ylabel('FRET Index')
    if stim_change:
        changes = [True if scF.stim[x] != scF.stim[x + 1] else False for x in range(len(scF.stim) - 1)]
        change_vals = [scF.Tt[x] for x in range(len(scF.stim) - 1) if changes[x]]
        for i in range(len(change_vals)):
            stimax.axvline(x=change_vals[i], color='black', linestyle='--', lw=1, alpha=1.0)
            fretax.axvline(x=change_vals[i], color='black', linestyle='--', lw=1, alpha=1.0)
    plt.show()
    plt.savefig('%s/%s.png' % (out_dir, spec_name))
    plt.close()


def plot_params(params, params_err, pnames, spec_name):
    if np.isnan(params_err).any():
        print("Parameters for spec %s do not have valid error bounds." % spec_name)
        return None

    fig, axs = plt.subplots(len(params), len(params), figsize=(15, 15))
    for i in range(len(params)):
        for j in range(len(params)):
            mu = np.asarray([params[i], params[j]])
            if i != j:
                sigma = np.asarray([[params_err[i, i], params_err[i, j]], [params_err[j, i], params_err[j, j]]])
                x, y, z = gauss(mu, sigma)
                axs[i, j].contour(x, y, z)
            else:
                mu = mu[0]
                std = np.sqrt(params_err[i, i])
                x, y, z = circle(mu, std)
                axs[i, j].contour(x, y, z, [1.0, 2.0, 3.0], colors=['green', 'orange', 'red'])
            axs[i, j].set_xlabel(pnames[i])
            axs[i, j].set_ylabel(pnames[j])
    out_dir = '%s/plots/%s' % (data_dir, spec_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig('%s/parameters.png' % out_dir)
    plt.close()

def parse_population_params(model, stim_prot, cell_names, plot=True, plot_cov=True, KDE=False):
    main_dir = data_dir
    assert isinstance(model, CellModel), "Model is not a CellModel object"
    ncells = len(cell_names)
    npars = len(model.params_set)
    pars = np.zeros((ncells, npars))
    pars_err = np.zeros((ncells, npars, npars))
    model_name = model.__class__.__name__

    for cell_no, cell in enumerate(cell_names):
        filename = "%s/results/%s_%s_%s.txt" % (main_dir, stim_prot, model_name, cell)
        try:
            with open(filename, 'r') as results_file:
                data = json.load(results_file)
        except:
            print("Cell %s had an error in parameter inference." % cell)
            pars[cell_no] = np.nan
            pars_err[cell_no] = np.nan
            continue
        pars[cell_no] = data['par']
        pars_err[cell_no] = data['par_err']

    valid_mask = np.full((len(pars),), True)
    for p in range(npars):
        valid_mask = np.logical_and(valid_mask, np.logical_and(pars[:, p] >= model.param_bounds[p][0],
                                                               pars[:, p] <= model.param_bounds[p][1]))
    valid_par = pars[valid_mask]
    valid_err = pars_err[valid_mask]
    print("Number of cells with valid parameter estimates is %s of %s" % (len(valid_par), ncells))

    err_mask = ~np.isnan(valid_err).any(axis=(1,2))
    err_par = valid_par[err_mask]
    err_err = valid_err[err_mask]
    print("Number of those cells with parameter error estimates is %s of %s" % (len(err_par), len(valid_par)))

    if plot:
        if KDE:
            fig, axs = plt.subplots(npars, figsize=(10, 15))
            for p in range(npars):
                sns.kdeplot(valid_par[:, p], ax=axs[p], shade=True)
                axs[p].set_xlabel(model.param_names[p])
            fig.suptitle('Parameters for Stimulus %s, Model %s' % (stim_prot, model_name))
            plt.savefig("%s/results/Pars_%s_%s_KDE" % (main_dir, stim_prot, model_name))
            plt.close()
        else:
            num_points = 1000
            fig, axs = plt.subplots(npars, figsize=(10, 15))
            for p in range(npars):
                domain = np.linspace(model.param_bounds[p][0], model.param_bounds[p][1], num_points)
                results = np.zeros_like(domain)
                for c in range(len(err_par)):
                    pdf = norm.pdf(domain, loc=err_par[c, p], scale=np.sqrt(err_err[c, p, p]))
                    results += pdf
                axs[p].fill_between(domain, results, alpha=0.3, facecolor='blue')
                axs[p].set_xlabel(model.param_names[p])
                axs[p].set_ylabel("PDF")
                axs[p].set_yticklabels([])
                axs[p].set_xlim(model.param_bounds[p][0], model.param_bounds[p][1])
            fig.suptitle('Parameters for Stimulus %s, Model %s' % (stim_prot, model_name))
            plt.savefig("%s/results/Pars_%s_%s" % (main_dir, stim_prot, model_name))
            plt.close()

    if plot_cov:
        if KDE:
            for p1 in range(npars):
                for p2 in range(p1):
                    ax = sns.jointplot(x=valid_par[:, p1], y=valid_par[:, p2], kind='reg')
                    p1name = model.param_names[p1]
                    p2name = model.param_names[p2]
                    ax.ax_joint.set_xlabel(p1name)
                    ax.ax_joint.set_ylabel(p2name)
                    plt.savefig('%s/results/ParCov_%s_%s_%s_%s_KDE' % (main_dir, stim_prot, model_name, p1name, p2name))
                    plt.close()
        else:
            for p1 in range(npars):
                for p2 in range(p1):
                    fig, axs = plt.subplots(2, 2, figsize=(8, 6),
                                            gridspec_kw={'hspace': 0, 'wspace': 0,
                                                         'width_ratios': [5, 1], 'height_ratios': [1, 5]})
                    axs[0, 0].axis("off")
                    axs[0, 1].axis("off")
                    axs[1, 1].axis("off")

                    domain1 = np.linspace(model.param_bounds[p1][0], model.param_bounds[p1][1], num_points)
                    results = np.zeros_like(domain1)
                    for c in range(len(err_par)):
                        pdf = norm.pdf(domain1, loc=err_par[c, p1], scale=np.sqrt(err_err[c, p1, p1]))
                        results += pdf
                    axs[0, 0].fill_between(domain1, results, alpha=0.5, facecolor='blue')
                    axs[0, 0].set_xlim(model.param_bounds[p1][0], model.param_bounds[p1][1])

                    domain2 = np.linspace(model.param_bounds[p2][0], model.param_bounds[p2][1], num_points)
                    results = np.zeros_like(domain2)
                    for c in range(len(err_par)):
                        pdf = norm.pdf(domain2, loc=err_par[c, p2], scale=np.sqrt(err_err[c, p2, p2]))
                        results += pdf
                    axs[1, 1].fill_betweenx(domain2, results, alpha=0.5, facecolor='blue')
                    axs[1, 1].set_ylim(model.param_bounds[p2][0], model.param_bounds[p2][1])

                    X, Y = np.meshgrid(domain1, domain2)
                    results = np.zeros_like(X)
                    for c in range(len(err_par)):
                        mean = np.array([err_par[c, p1], err_par[c, p2]])
                        sigma = np.array([[err_err[c, p1, p1], err_err[c, p1, p2]], [err_err[c, p2, p1], err_err[c, p2, p2]]])
                        pos = np.dstack((X, Y))
                        rv = multivariate_normal(mean, sigma)
                        z = rv.pdf(pos)
                        results += z
                    axs[1, 0].pcolormesh(X, Y, results, shading='nearest')
                    axs[1, 0].set_xlim(model.param_bounds[p1][0], model.param_bounds[p1][1])
                    axs[1, 0].set_ylim(model.param_bounds[p2][0], model.param_bounds[p2][1])
                    p1name = model.param_names[p1]
                    p2name = model.param_names[p2]
                    axs[1, 0].set_xlabel(p1name)
                    axs[1, 0].set_ylabel(p2name)
                    fig.suptitle('Parameters %s and %s for Stimulus %s, Model %s' % (p1name, p2name,
                                                                                     stim_prot, model_name))
                    plt.savefig('%s/results/ParCov_%s_%s_%s_%s' % (main_dir, stim_prot, model_name, p1name, p2name))
                    plt.close()
    return err_par, err_err
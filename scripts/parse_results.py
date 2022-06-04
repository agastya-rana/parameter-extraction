import sys
sys.path.append('..')
sys.path.append('../src')
import numpy as np
import json
import itertools
from scipy.stats import norm, multivariate_normal
from src.local_methods import def_data_dir
import matplotlib.pyplot as plt
from src.models import *
import seaborn as sns
main_dir = def_data_dir()

num_cells = 29
model = MWC_MM_Swayam()
#batches = [220517, 220518]
batches = [220517]
cells = [i for i in range(num_cells)]
stim_prot = "BS"
cell_names = ["%s_%s" % (batch, cell) for batch, cell in itertools.product(batches, cells)]

def parse_population_params(model, stim_prot, cell_names, plot=True, plot_cov=True, KDE=False):
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

parse_population_params(model, stim_prot, cell_names, KDE=True)
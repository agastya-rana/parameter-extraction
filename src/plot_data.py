"""
Plot stimulus/measurement data and prediction data.

Created by Nirag Kadakia and Agastya Rana, 11/18/21.
"""

import os
import scipy as sp
from single_cell_FRET import single_cell_FRET
from load_data import load_stim_file, load_meas_file
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
from local_methods import def_data_dir
data_dir = def_data_dir()

def plot_trajectories(spec_name, scF, est_path=None, pred_path=None, plot_observed=False):
    # Load all of the prediction data and estimation object and dicts
    full_range = sp.arange(scF.est_wind_idxs[0], scF.pred_wind_idxs[-1])
    est_Tt = scF.Tt[scF.est_wind_idxs]
    pred_Tt = scF.Tt[scF.pred_wind_idxs]
    full_Tt = scF.Tt[full_range]

    fig, axs = plt.subplots(scF.nD + 1, 1, sharex=True)
    # Plot the stimulus
    axs[0].plot(full_Tt, scF.stim[full_range], color='r', lw=2)
    axs[0].set_xlim(full_Tt[0], full_Tt[-1])
    axs[0].set_ylim(80, 160)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Stimulus ($\mu$M)")

    plot_no = 1
    for num in range(scF.nD):
        if plot_observed and num not in scF.L_idxs:
            continue
        axs[plot_no].set_ylabel(scF.model.state_names[num])
        if num in scF.L_idxs:
            ## Plot Measured Data
            axs[num + 1].plot(full_Tt, scF.meas_data[full_range, num], color='g', label='Measured')
        ## Plot Inferred Data
        if est_path != None:
            axs[plot_no].plot(est_Tt, est_path[:, num], color='r', lw=1, label='Estimated')
        if pred_path != None:
            axs[plot_no].plot(pred_Tt, pred_path[:, num], color='k', lw=1, label='Predicted')
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
    out_dir = '%s/plots/%s' % (DATA_DIR, spec_name)
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


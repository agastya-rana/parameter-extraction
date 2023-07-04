import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

## Function to get data from excel files
def get_data(filename, ncells):
    ## Get data from excel file
    ## First column is time, second column should be ignored, read all other columns
    data = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(0,)+tuple(range(2, ncells+2)))
    return data


def compute_autocorr_cells(data, tau_max):
    print(data.shape)
    ## Compute autocorrelation for each cell
    taus = []
    autocorrs = []
    for i in range(1, data.shape[1]):
        ## Get timestamps and measurements for cell i
        timestamps = data[:, 0]
        measurements = data[:, i]
        ## Compute autocorrelation
        tau, autocorr = compute_autocorrelation(timestamps, measurements, tau_max)
        taus.append(tau)
        autocorrs.append(autocorr)
    return taus, autocorrs

def compute_autocorrelation(timestamps, measurements, tau_max):
    ## Computes autocorrelation of measurements of uneven sampling_intervals
    sample_interval, switches = np.unique(np.diff(timestamps), return_index=True)
    ## Break the measurements into chunks of equal sampling intervals
    measurement_blocks = np.split(measurements, switches[1:])
    mean = np.mean(measurements)
    ## Figure out the values of tau to compute autocorrelation for
    ## Can compute any multiple of any sample_interval, as long as it is less than tau_max
    taus = []
    for i in range(len(switches)):
        taus += list(sample_interval[i] * np.arange(1, int(tau_max/sample_interval[i])))
    taus = np.array(taus)
    ## Sort taus
    taus = np.sort(taus)
    ## Compute autocorrelation
    autocorr = np.zeros(len(taus))
    for i, tau in enumerate(taus):
        ## Figure out which sample intervals can be used
        interval_mask = (tau % sample_interval) == 0
        ## Start points include every point with a sampling rate that can be used minus those near the end
        j = 1
        x = measurement_blocks[j][:-int(tau//sample_interval[j])]
        start_points = np.concatenate([measurement_blocks[j][:-int(tau//sample_interval[j])] for j, mask in enumerate(interval_mask) if mask])
        ## End points are the start points shifted by tau
        end_points = np.concatenate([measurement_blocks[j][int(tau//sample_interval[j]):] for j, mask in enumerate(interval_mask) if mask])
        autocorr[i] = np.mean((start_points - mean) * (end_points - mean))
    autocorr /= np.mean((measurements-mean)**2)
    return taus, autocorr

def plot_autocorrelations(filenames, ncells=183, tau_max=50, filename_titles=None):
    ## Each filename has a different kind of autocorrelation
    all_taus = []
    all_autocorrs = []
    for filename in filenames:
        ## Get data from excel file
        data = get_data(filename, ncells=ncells)
        ## Compute autocorrelation for each cell
        taus, autocorrs = compute_autocorr_cells(data, tau_max=tau_max)
        all_taus.append(taus)
        all_autocorrs.append(autocorrs)
    ## Plot autocorrelation for each cell
    for i in range(ncells):
        ## Make one subplot for each filename
        fig, axes = plt.subplots(1, len(filenames), sharex=True, sharey=True, figsize=(10*len(filenames), 10))
        for j, filename in enumerate(filenames):
            axes[j].plot(all_taus[j][i], all_autocorrs[j][i])
            axes[j].set_title(filename_titles[j] if filename_titles is not None else filename)
            axes[j].set_xlabel('tau')
            axes[j].set_ylabel('autocorrelation')
            axes[j].set_xlim([0, tau_max])
            axes[j].set_ylim([-1, 1])
        plt.savefig('autocorrelation_cell_%d.png' % i)
        plt.close()
    ## Convert to numpy arrays
    all_taus = np.array(all_taus)
    all_autocorrs = np.array(all_autocorrs)
    print(all_taus.shape, all_autocorrs.shape)
    return all_taus, all_autocorrs

def plot_avg_autocorr(taus, autocorrs):
    ## Plot average autocorrelation
    avg_autocorr = np.mean(autocorrs, axis=1)
    std_autocorr = np.std(autocorrs, axis=1)
    taus = taus[:, 0, :]
    print(taus.shape, avg_autocorr.shape, std_autocorr.shape)
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10*2, 10))
    for j in range(2):
        axes[j].errorbar(taus[j], avg_autocorr[j], yerr=std_autocorr[j])
        axes[j].set_xlabel('tau')
        axes[j].set_ylabel('autocorrelation')
        axes[j].set_xlim([0, 30])
        axes[j].set_ylim([-1, 1])
    plt.savefig('avg.png')
    plt.close()

def fit_exponential_decay(tau, autocorr):
    def exponential_decay(t, tau, scale):
        return scale * np.exp(-t / tau)

    popt, _ = curve_fit(exponential_decay, tau, autocorr)
    timescale = popt[0]
    scale = popt[1]

    return timescale, scale

if __name__ == '__main__':
    ## Plot autocorrelation for each cell
    filenames = ['Three_segments_model_residual_timeseries_183.csv', 'Three_segments_repeat_average_residual_timeseries_183.csv']
    filename_titles = ['Model Residual Autocorr', 'Average Residual Autocorr']
    x,y = plot_autocorrelations(filenames, filename_titles=filename_titles, tau_max=30, ncells=183)
    plot_avg_autocorr(x, y)
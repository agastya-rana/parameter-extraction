import numpy as np
## Import plotting library
import matplotlib.pyplot as plt

## function to generate an OU process
def OU_process(tau, sigma, dt, N):
    ## tau is the timescale of the process
    ## sigma is the standard deviation of the process
    ## dt is the timestep
    ## N is the number of timesteps
    ## returns an array of length N
    ## initialize array
    x = np.zeros(N)
    ## set first entry to 0
    x[0] = 0
    ## loop through the rest of the entries
    for i in range(1, N):
        ## set the entry to the previous entry minus the previous entry divided by tau times dt plus a random number drawn from a normal distribution with mean 0 and standard deviation sigma * sqrt(dt)
        x[i] = x[i-1] - x[i-1] / tau * dt + np.random.normal(0, sigma * np.sqrt(dt))
    ## return the array
    return x

## Compute autocorrelation
def compute_autocorr(x):
    ## Compute mean
    mean = np.mean(x)
    ## Compute autocorrelation
    autocorr = np.zeros(len(x))
    for i in range(len(x)):
        autocorr[i] = np.mean((x[:len(x)-i]-mean)*(x[i:]-mean))
    autocorr /= np.mean((x-mean)**2)
    return autocorr

## Segment x into segments of length L, compute autocorrelation for each segment, and average the autocorrelations
def compute_avg_autocorr(x, L):
    ## Compute number of segments
    nsegments = int(len(x) / L)
    ## Initialize array of autocorrelations
    autocorrs = np.zeros((nsegments, L))
    ## Loop through segments
    for i in range(nsegments):
        ## Get segment
        segment = x[i*L:(i+1)*L]
        ## Compute autocorrelation
        autocorrs[i] = compute_autocorr(segment)
    ## Average autocorrelations
    avg_autocorr = np.mean(autocorrs, axis=0)
    return avg_autocorr

## Plot both estimates of autocorrelation on the same plot
def plot_autocorrs(x, L, tau, sigma, dt, N):
    ## Compute autocorrelation
    autocorr = compute_autocorr(x)
    ## Compute average autocorrelation
    avg_autocorr = compute_avg_autocorr(x, L)
    ## Plot autocorrelations
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10*2, 10))
    axes[0].plot(np.arange(len(autocorr)) * dt, autocorr)
    axes[0].set_xlabel('tau')
    axes[0].set_ylabel('autocorrelation')
    axes[0].set_xlim([0, tau])
    axes[0].set_ylim([-1, 1])
    axes[0].set_title('Autocorrelation')
    axes[1].plot(np.arange(len(avg_autocorr)) * dt, avg_autocorr)
    axes[1].set_xlabel('tau')
    axes[1].set_ylabel('autocorrelation')
    axes[1].set_xlim([0, tau])
    axes[1].set_ylim([-1, 1])
    axes[1].set_title('Average Autocorrelation')
    plt.savefig('autocorrelation.png')
    plt.close()


## Plot the power spectrum of the OU process on a log log plot in both ways (raw, and then by averaging the power spectrum of segments of length L)
def plot_power_spectrum(x, dt, N):
    ## Compute power spectrum
    power_spectrum = np.abs(np.fft.fft(x))**2
    ## Compute frequencies
    freqs = np.fft.fftfreq(len(x), dt)
    ## Plot power spectrum
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10*2, 10))
    axes[0].loglog(freqs, power_spectrum)
    axes[0].set_xlabel('frequency')
    axes[0].set_ylabel('power')
    axes[0].set_xlim([0, 1/dt/2])
    axes[0].set_title('Power Spectrum')
    ## Compute average power spectrum
    L = 1000
    nsegments = int(len(x) / L)
    power_spectrums = np.zeros((nsegments, L))
    for i in range(nsegments):
        segment = x[i*L:(i+1)*L]
        power_spectrums[i] = np.abs(np.fft.fft(segment))**2
    avg_power_spectrum = np.mean(power_spectrums, axis=0)
    ## Plot average power spectrum
    axes[1].loglog(freqs[:L], avg_power_spectrum)
    axes[1].set_xlabel('frequency')
    axes[1].set_ylabel('power')
    axes[1].set_xlim([0, 1/dt/2])
    axes[1].set_title('Average Power Spectrum')
    plt.savefig('power_spectrum.png')
    plt.close()

if __name__ == '__main__':

    ## Set random seed
    np.random.seed(0)
    ## Set parameters
    tau = 1
    sigma = 1
    dt = 0.01
    N = 100000
    L = 1000
    ## Generate OU process
    x = OU_process(tau, sigma, dt, N)
    ## Plot autocorrelations
    plot_autocorrs(x, L, tau, sigma, dt, N)
    ## Plot power spectrum
    plot_power_spectrum(x, dt, N)


"""
General, miscellaneous functions for FRET data assimilation.

Nirag Kadakia & Agastya Rana, 11/12/21.
"""

import scipy as sp
from scipy.ndimage.filters import gaussian_filter
import json
from src.local_methods import def_data_dir


def noisify(Ss, params=[0, 1]):
    """
	Adds Gaussian noise to any vector.
	"""

    mu, sigma = params
    size = Ss.shape
    Ss += sp.random.normal(mu, sigma, size)
    return Ss


def smooth_vec(x, window_len=11, window='gaussian'):
    """
	Smooth 1D data using a window with requested size. This method is 
	based on the convolution of a scaled window with the signal.
	The signal is prepared by introducing reflected copies of the signal 
	(with the window size) in both ends so that transient parts are minimized
	in the begining and end part of the output signal.
	
	input:
		x: the input signal 
		window_len: the dimension of the smoothing window; 
					should be an odd integer
		window: the type of window from 'flat', 'hanning', 
				'hamming', 'bartlett', 'blackman'. flat window 
				will produce a moving average smoothing.
	output:
		the smoothed signal

	example:

	t=linspace(-2,2,0.1)
	x=sin(t)+randn(len(t))*0.1
	y=smooth(x)
	
	NOTE: length(output) != length(input), to correct this: 
			return y[(window_len/2-1):-(window_len/2)] instead of just y.
	"""

    if window == 'gaussian':
        x = gaussian_filter(x, sigma=window_len)
        return x

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning'," + "'hamming', 'bartlett', 'blackman'")

    s = sp.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = sp.ones(window_len, 'd')
    else:
        w = eval('sp.' + window + '(window_len)')

    y = sp.convolve(w / w.sum(), s, mode='valid')
    return y[:len(x)]


def create_sample_json(filename):
    data_dir = def_data_dir()
    data_vars = {'nD': 2, 'nT': 500, 'nP': 7, 'dt': 0.5, 'stim_type': 'block', 'stim_params': [0.01],
                 'meas_noise': [0.01], 'L_idxs': 1}
    est_vars = {'model': 'MWC_linear', 'params_set': [20., 3225., 0.5, 2.0, 6.0, 0.33, -0.01], 'bounds_set': 'default',
                'est_beg_T': 0, 'est_end_T': 500, 'pred_end_T': 500}
    est_specs = {'est_type': 'VA'}
    data_dict = {'data_vars': data_vars, 'est_vars': est_vars, 'est_specs': est_specs}
    filepath = '%s/specs/%s' % (data_dir, filename)
    with open(filepath, 'w') as outfile:
        json.dump(data_dict, outfile, indent=4)
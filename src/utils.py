"""
General, miscellaneous functions for FRET data assimilation

Created by Nirag Kadakia at 12:30 10-16-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
from scipy.ndimage.filters import gaussian_filter
import sys


def get_flags():
    """
	Args:
		arg_num: the command line argument number
	
	Returns:
		data_flag: string
	"""

    data_flags = []
    if len(sys.argv) < 2:
        raise Exception("Need at least 1 tag for the data in command line")
        quit()
    else:
        for flag in sys.argv[1:]:
            data_flags.append(str(flag))
    return data_flags


def get_flag(arg_num=1):
    """
	Args:
		arg_num: the command line argument number
	
	Returns:
		data_flag: string
	"""
    try:
        data_flag = str(sys.argv[arg_num])
    except:
        raise Exception("Need to specify a tag for the data in command line")

    return data_flag


def merge_two_dicts(x, y):
    """
	Given two dicts, merge them into a	new dict as a shallow copy.
	"""

    z = x.copy()
    z.update(y)

    return z


def noisify(Ss, params=[0, 1]):
    """
	Adds noise to any vector.
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
        raise ValueError("Window is one of 'flat', 'hanning'," \
                         "'hamming', 'bartlett', 'blackman'")

    s = sp.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = sp.ones(window_len, 'd')
    else:
        w = eval('sp.' + window + '(window_len)')

    y = sp.convolve(w / w.sum(), s, mode='valid')
    return y[:len(x)]

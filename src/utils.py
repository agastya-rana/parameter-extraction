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
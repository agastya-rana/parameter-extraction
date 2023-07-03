"""
General, miscellaneous functions for FRET data assimilation.

Nirag Kadakia & Agastya Rana, 11/12/21.
"""

import scipy as sp
import scipy.stats
import numpy as np
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

def gauss(mu, sigma):
    n = 1000
    xstd = np.sqrt(sigma[0,0])
    ystd = np.sqrt(sigma[1,1])
    xmean = mu[0]
    ymean = mu[1]
    x = np.linspace(xmean - 3 * xstd, xmean + 3 * xstd, n)
    y = np.linspace(ymean - 3 * ystd, ymean + 3 * ystd, n)
    xmesh, ymesh = np.meshgrid(x, y)
    pos = np.dstack((xmesh, ymesh))
    rv = scipy.stats.multivariate_normal(mu, sigma)
    z = rv.pdf(pos)
    return xmesh, ymesh, z

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def circle(mu, std):
    n = 1000
    x = np.linspace(mu - 4 * std, mu + 3 * std, n)
    xmesh, ymesh = np.meshgrid(x, x)
    pos = np.dstack((xmesh, ymesh))
    z = np.sqrt(np.square(xmesh-mu)+np.square(ymesh-mu))/std
    return xmesh, ymesh, z

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
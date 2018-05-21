"""
Linear kernel estimation of twin FRET data

Created by Nirag Kadakia at 15:20 10-25-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys, time
sys.path.append('../src')
import scipy as sp
from utils import get_flags
from single_cell_FRET import single_cell_FRET
from load_data import load_VA_twin_data
from save_data import save_estimated_kernels
from params_bounds import *


def single_cell_FRET_kernel(data_flags, 
			log_regularizations=sp.linspace(-3, 3, 10)):
	"""
	Function to estimate a linear kernel on FRET data. 
	Uses data saved from MWC_twin_data.py; run this first.
	command line arguments: data_flag, dt, data noise, and 
	regularization parameter for least squares optimization
	"""	

	data_ID = data_flags[0]
	data_dt = float(data_flags[1])
	data_sigma = float(data_flags[2])
	kernel_length = int(data_flags[3])
	regularizations = 10.**log_regularizations

	scF = single_cell_FRET()

	data_dict = load_VA_twin_data(data_flags=data_flags)
	measurements = data_dict['measurements'][:, 1:]
	scF.signal_vector = data_dict['stimuli'][:]
	scF.Tt = data_dict['measurements'][:, 0]
	scF.nT = len(scF.Tt)
	scF.dt = scF.Tt[1]-scF.Tt[0]

	scF.kernel_length = kernel_length
	scF.set_kernel_estimation_Aa()
	estimated_kernels = sp.zeros((scF.kernel_length, len(regularizations)))

	for iR, regularization in enumerate(regularizations):
		scF.regularization = regularization
		scF.set_kernel_estimation_regularization()
		scF.set_kernel_estimation_inverse_hessian()
		kernel_response_vector = measurements[scF.kernel_length:, 0]
		estimated_kernels[:, iR] = scF.kernel_calculation(kernel_response_vector)
	
	save_estimated_kernels(estimated_kernels, data_flags)


if __name__ == '__main__':
	data_flags = get_flags()
	single_cell_FRET_kernel(data_flags)
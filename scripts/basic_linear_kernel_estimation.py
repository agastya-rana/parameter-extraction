"""
Tikhonov regularization to get linear kernel

Created by Nirag Kadakia at 22:20 10-11-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import sys
from scipy import signal
import scipy.linalg as LA
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import matplotlib.pyplot as plt
sys.path.append('../src')
from load_data import load_preliminary_FRET

data = load_preliminary_FRET(data_set=1, cell=12)

files = sp.arange(20, 50)
loaded_files = 0
FRET_data = []
loaded_files = []

for cell in files:
	try:
		data = load_preliminary_FRET(data_set=1, cell=cell)
		FRET_data_vec = data['FRET_idx']
		FRET_data_vec = FRET_data_vec
		FRET_data_vec = FRET_data_vec - sp.average(FRET_data_vec)
		FRET_data_vec = -FRET_data_vec
		FRET_data.append(FRET_data_vec)
		loaded_files.append(cell)
	except:
		print ('%s not loaded' % cell)


sig_l_lim = 150
sig_u_lim = 640
kernel_length = 150
lag = 50
dt = 0.1

Xx = data['signal'][sig_l_lim:sig_u_lim]
Xx = Xx - sp.average(Xx)
signal_length = len(Xx)
kernel_x = sp.arange(-lag*dt, (kernel_length - lag)*dt, dt)

fig = plt.figure()
fig.set_size_inches(3, 5)

for reg_lambda in 10.**sp.arange(4.0, 5.0, 10.0):
	
	for cell_idx, cell in enumerate(loaded_files):
		Aa = sp.zeros((signal_length - kernel_length, kernel_length))
		for row, val in enumerate(Aa[:, 0]):
			Aa[row, :] = Xx[row: row + kernel_length][::-1]
		lambda_squared = sp.eye(kernel_length)*reg_lambda**2.0
		
		Yy = FRET_data[cell_idx][sig_l_lim + kernel_length - lag:(sig_u_lim - lag)]
		
		inverse_hessian = LA.inv(sp.dot(Aa.T, Aa) + lambda_squared)
		kernel_est = sp.dot(sp.dot(inverse_hessian, Aa.T), Yy)

		plt.subplot(111)
		plt.plot(kernel_x, kernel_est, linewidth=0.3)

	FRET_data = sp.array(FRET_data)
	Yy = sp.average(FRET_data, axis=0)
	Yy = Yy[sig_l_lim + kernel_length - lag:(sig_u_lim - lag)]

	avg_kernel_est = sp.dot(sp.dot(inverse_hessian, Aa.T), Yy)
	plt.subplot(111)
	plt.plot(kernel_x, avg_kernel_est, linestyle = '--')

plt.show()



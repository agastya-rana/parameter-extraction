"""
Plot activity averages and deviations over experiments

Created by Nirag Kadakia at 14:40 10-16-2017
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

files = sp.arange(0, 60)
loaded_files = 0
FRET_data = []
loaded_files = []

for cell in files:
	try:
		data = load_preliminary_FRET(data_set=1, cell=cell)
		FRET_data_vec = data['FRET_idx']
		FRET_data.append(FRET_data_vec)
		loaded_files.append(cell)
	except:
		print ('%s not loaded' % cell)

FRET_data = sp.array(FRET_data)

sig_l_lim = 0
sig_u_lim = 700
dt = 0.5
Tt = sp.arange(sig_l_lim*dt, sig_u_lim*dt, dt)

fig = plt.figure()
fig.set_size_inches(3, 5)

# Stats
FRET_data_avg = sp.average(FRET_data[:, sig_l_lim: sig_u_lim], axis = 0)
FRET_data_std = sp.zeros(len(Tt))
for iT in range(len(Tt)):
	FRET_data_std[iT] = sp.std(FRET_data[:, iT], axis = 0)
z_critical = 1.95
margin_of_error = z_critical*(FRET_data_std/sp.sqrt(len(loaded_files)))
confidence_interval = sp.array([FRET_data_avg - margin_of_error, FRET_data_avg + margin_of_error])
y_err = (confidence_interval[1, :] - confidence_interval[0, :])/2

plt.subplot(211)
for cell_idx, cell in enumerate(loaded_files):
	Yy = FRET_data[cell_idx, sig_l_lim: sig_u_lim]
	plt.plot(Tt, Yy, linewidth=0.2)
plt.errorbar(x=Tt, 
             y=FRET_data_avg, 
             yerr=y_err,
             fmt='o')   
plt.axhline(sp.average(FRET_data_avg), xmin=0, xmax=Tt[-1])

plt.subplot(212)
plt.plot(Tt, data['signal'][sig_l_lim: sig_u_lim])
plt.ylim(50, 150)
plt.show()

print '\nSignal stats:'
print sp.average(data['signal'][200: 500])
print sp.std(data['signal'][200: 500])

print '\nAverage maximum values for each cell:'
print sp.average(sp.amax(FRET_data, axis = 1))

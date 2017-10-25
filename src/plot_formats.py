"""
Functions for generating plot formats for various types of plots.

Created by Nirag Kadakia at 21:46 10-26-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
from local_methods import def_data_dir
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
import matplotlib.pyplot as plt

					
DATA_DIR = def_data_dir()

def opt_kernel_pred_plot(dt, kernel_length, Tt_PW):

	fig = plt.figure()
	fig.set_size_inches(6, 6)	
	
	plt.subplot(311)
	plt.title(r'Mean subtracted stimulus')
	plt.xlabel(r'Time (s)')
	plt.ylabel(r'$\mu m$')
	plt.xlim(Tt_PW[0], Tt_PW[-1])
	
	plt.subplot(312)	
	plt.title(r'Mean subtracted FRET prediction')
	plt.xlabel(r'Time (s)')
	plt.xlim(Tt_PW[0], Tt_PW[-1])
	
	plt.subplot(313)	
	plt.title(r'Estimated Kernel')
	plt.xlabel(r'Time (s)')
	plt.ylabel(r'$1/\mu m$')
	plt.xlim(0, dt*kernel_length)
	
	return fig
	

def kernel_pred_plot(dt, kernel_length, Tt_EW, Tt_PW):

	fig = plt.figure()
	fig.set_size_inches(6, 6)	
	
	plt.subplot(311)
	plt.title(r'Mean subtracted stimulus')
	plt.xlabel(r'Time (s)')
	plt.ylabel(r'$\mu m$')
	plt.xlim(Tt_EW[0], Tt_PW[-1])
	
	plt.subplot(312)	
	plt.title(r'Mean subtracted optimal FRET prediction')
	plt.xlabel(r'Time (s)')
	plt.xlim(Tt_EW[0], Tt_PW[-1])
	plt.axvline(Tt_EW[-1], ymin=0, ymax=1, color='yellow', lw=0.5)
	
	plt.subplot(313)	
	plt.title(r'Estimated Kernel')
	plt.xlabel(r'Time (s)')
	plt.ylabel(r'$1/\mu m$')
	plt.xlim(0, dt*kernel_length)
	
	return fig
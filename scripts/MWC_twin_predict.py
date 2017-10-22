"""
Prediction plotting from MWC estimates.

Created by Nirag Kadakia at 17:00 10-22-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys, time
sys.path.append('../src')
import scipy as sp
import matplotlib.pyplot as plt
from utils import get_flags
from single_cell_FRET import single_cell_FRET


def plot_prediction(data_flags):
	"""
	TODO
	"""	
	
	a = single_cell_FRET()

	a.dt = float(data_flags[1])
	FRET_noise = float(data_flags[2])

	a.set_Tt() 
	a.

	a.df_integrate()
	
	save_twin_data(a.Tt, a.true_states, a.signal_vector, 
					measured_vars_and_noise=[[1, FRET_noise]], 
					data_flags=data_flags)
	
if __name__ == '__main__':
	data_flags = get_flags()
	generate_MWC_twin_data(data_flags)							

"""
Twin data generation of MWC model and save to file

Created by Nirag Kadakia at 08:00 10-16-2017
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
from models import MWC_Tar
from save_data import save_twin_data


def generate_MWC_twin_data(data_flags, x0 = sp.array([1.27, 7.0])):
	"""
	Function to generate twin data for the MWC model. Command line
	arguments: data_flag, dt, data noise.
	"""

	a = single_cell_FRET()	

	a.dt = float(data_flags[1])
	FRET_noise = float(data_flags[2])
	print ('Setting dt=%s and noise=%s' % (a.dt, FRET_noise))
	a.x_integrate_init = x0

	a.set_Tt() 
	a.set_step_signal()
	a.model = MWC_Tar
	a.set_true_params()
	a.df_integrate()
	
	save_twin_data(a.Tt, a.true_states, a.signal_vector, 
					measured_vars_and_noise=[[1, FRET_noise]], 
					data_flags=data_flags)
	
if __name__ == '__main__':
	data_flags = get_flags()
	generate_MWC_twin_data(data_flags)							

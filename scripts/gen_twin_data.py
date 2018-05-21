"""
Twin FRET data generation and also fake stimuli, if desired.

Created by Nirag Kadakia at 08:00 10-16-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys
sys.path.append('../src')
import scipy as sp
from utils import get_flag
from load_specs import read_specs_file, compile_all_run_vars
from save_data import save_stim, save_true_states, save_meas_data, \
						save_stim_and_meas_plots
from single_cell_FRET import single_cell_FRET


def gen_twin_data(data_flag):
	
	# Load specifications from file; to be passed to single_cell_FRET object
	list_dict = read_specs_file(data_flag)
	vars_to_pass = compile_all_run_vars(list_dict)
	scF = single_cell_FRET(**vars_to_pass)
	
	assert scF.meas_file is None, "For generating twin data manually, cannot "\
		"import a measurement file; remove meas_file var in specs file"
	
	scF.set_stim()
	scF.gen_true_states()
	scF.set_meas_data()
	
	save_stim(scF, data_flag)
	save_true_states(scF, data_flag)
	save_meas_data(scF, data_flag)
	save_stim_and_meas_plots(scF, data_flag)
	
	
if __name__ == '__main__':
	data_flag = get_flag()
	gen_twin_data(data_flag)
"""
Generate a python dataset from the saved matlab structure of FRET recordings.

Created by Nirag Kadakia at 08:00 10-16-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys
import os
scripts_path = [i for i in sys.path if 'scripts' in i][0]
sys.path.append(os.path.join(os.path.dirname(scripts_path),'src'))
import scipy as sp
from load_data import load_FRET_recording
from save_data import save_stim, save_meas_data
from single_cell_FRET import single_cell_FRET


def gen_py_data_from_mat(dir='170913/Device1/FRET1', 
							mat_file='FRET_data_workspace', cell=17):
	
	data = load_FRET_recording(dir, cell=cell, mat_file=mat_file)
	
	a = single_cell_FRET()
	a.stim = data['stim']
	a.Tt = data['Tt']
	a.meas_data = data['FRET_idx']
	
	data_flag = '%s_cell_%s' % (dir.replace('/', '_'), cell)
	save_stim(a, data_flag)
	save_meas_data(a, data_flag)


if __name__ == '__main__':
	gen_py_data_from_mat()
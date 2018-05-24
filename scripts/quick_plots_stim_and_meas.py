"""
Quick plots of all stimuli and measurements for which data
exists in the stmi/ and meas/ folders

Created by Nirag Kadakia at 14:45 05-21-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import os
import sys
sys.path.append('../src')
from local_methods import def_data_dir
from load_data import load_stim_file, load_meas_file
from save_data import save_meas_plots, save_stim_plots
from single_cell_FRET import single_cell_FRET


print 'Plotting stimuli...'

stim_path = '%s/stim' % def_data_dir()
data_flags = []
for (dirpath, dirnames, filenames) in os.walk(stim_path):
	for filename in filenames:
		if filename.endswith('.stim'):
			data_flags.append(os.path.splitext(filename)[0])

for data_flag in data_flags:
	
	stim_Tt = load_stim_file(data_flag)
	scF = single_cell_FRET()
	scF.Tt = stim_Tt[:, 0]
	scF.stim = stim_Tt[:, 1:]
	save_stim_plots(scF, data_flag)

	
print 'Plotting measurements...'

meas_path = '%s/meas_data' % def_data_dir()
data_flags = []
for (dirpath, dirnames, filenames) in os.walk(meas_path):
	for filename in filenames:
		if filename.endswith('.meas'):
			data_flags.append(os.path.splitext(filename)[0])

for data_flag in data_flags:
	
	meas_Tt = load_meas_file(data_flag)
	scF = single_cell_FRET()
	scF.Tt = meas_Tt[:, 0]
	scF.meas_data = meas_Tt[:, 1:]
	save_meas_plots(scF, data_flag)
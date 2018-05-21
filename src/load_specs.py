"""
Module for reading and parsing specifications file for FRET data assimilation.

Created by Nirag Kadakia at 9:30 05-18-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import os
from local_methods import def_data_dir
from utils import merge_two_dicts

data_dir = def_data_dir()

def read_specs_file(data_flag, data_dir=data_dir):
	""" 
	Function to read a specifications file.
	
	Module to gather information from specifications file about how a 
	particular 	run is to be performed for FRET fake data generation
	and FRET data assimilation. 
	Specs file should have format .txt and the format is as listed here:

	
	data_var	       nT			        3
	est_var            set_param_bounds     bounds_Tar_3
	est_spec           est_type             VA
	
	It accepts these 3 types of inputs, labeled by the first column: variables
	relevant to data input and outpu and generation, those relevant to 
	estimation, and the type of estimation to be done.
	
	Args: 
		data_flag: Name of specifications file.
		data_dir: Data folder, if different than in local_methods.
	
	Returns:
		list_dict: Dictionary of 4 items keyed by 'data_vars', 
					'est_vars', and 'est_specs'.	

	"""

	filename = '%s/specs/%s.txt' % (data_dir, data_flag)	
	try:
		os.stat(filename)
	except:
		print ("There is no input file %s/specs/%s.txt" 
				% (data_dir, data_flag))
		exit()
	specs_file = open(filename, 'r')

	data_vars = dict()
	est_vars = dict()
	est_specs = dict()
	
	line_number = 0
	for line in specs_file:
		line_number += 1
		if line.strip():
			if not line.startswith("#"):
				keys = line.split()
				var_type = keys[0]
				if var_type == 'data_var':
					var_name = keys[1]
					try:
						data_vars[var_name] = float(keys[2])
					except ValueError:
						data_vars[var_name] = str(keys[2])
				elif var_type == 'est_var':
					var_name = keys[1]
					try:
						est_vars[var_name] = float(keys[2])
					except ValueError:
						est_vars[var_name] = str(keys[2])
				elif var_type == 'est_spec':
					var_name = keys[1]
					est_specs[var_name] = keys[2:]
				else:
					print ("Unidentified input on line %s of %s.txt: %s" 
							%(line_number, data_flag, line))
					quit()
		
	specs_file.close()
	print ('\n -- Input vars and params loaded from %s.txt\n' % data_flag)
	
	list_dict =  dict()
	for i in ('data_vars', 'est_vars', 'est_specs'):
		list_dict[i] = locals()[i]
	
	return list_dict
	
def compile_all_run_vars(list_dict):
	"""
	Grab all the run variables from the specifications file, and aggregate
	as a complete dictionary of variables to be specified and/or overriden
	in the single_cell_FRET_VA object.
	
	Args:
		list_dict: dictionary containing at least 2 keys; data_vars and 
			est_vars. These are read through read_specs_file()
			function in this module. Only these two keys are read
			in this module; other keys may exist, but will be ignored.
		
	Returns:
		vars_to_pass: dictionary whose keys are all variables to be overriden
			in the single_cell_FRET_VA class when initialized.
	"""
	
	vars_to_pass = dict()
	vars_to_pass = merge_two_dicts(vars_to_pass, list_dict['data_vars'])
	vars_to_pass = merge_two_dicts(vars_to_pass, list_dict['est_vars'])
	
	return vars_to_pass
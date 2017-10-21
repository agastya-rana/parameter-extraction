"""
Fixed parameter sets for importing.

Created by Nirag Kadakia at 15:40 10-17-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

from collections import OrderedDict

def params_Tar_1():
	
	params = OrderedDict()
	
	params['K_off_a'] = 0.02
	params['K_on_a'] = 0.5
	params['Nn'] = 5
	params['alpha_m'] = 2
	params['m_0'] = 0.5
	params['a_0'] = 0.33
	params['tau_m'] = 35.
	params['k_FR'] = 40.
	params['tau_FR'] = 0.5
	
	return params
	
	
def bounds_Tar_1():
	
	bounds = OrderedDict()
	
	bounds['K_off_a'] = [0.01, 0.05]
	bounds['K_on_a'] = [0.2, 0.7]
	bounds['Nn'] = [4, 5]
	bounds['alpha_m'] = [0.5, 10]
	bounds['m_0'] = [0.5, 0.5]
	bounds['a_0'] = [1e-3, 1e0]
	bounds['tau_m'] = [1, 200]
	bounds['k_FR'] = [1, 200]
	bounds['tau_FR'] = [0.01, 5]

	return bounds	

def bounds_Tar_2():

	bounds = OrderedDict()

	bounds['K_off_a'] = [0.018, 0.022]
	bounds['K_on_a'] = [0.45, 0.55]
	bounds['Nn'] = [4.5, 5.5]
	bounds['alpha_m'] = [1.8, 2.1]
	bounds['m_0'] = [0.45, 0.55]
	bounds['a_0'] = [0.31, 0.35]
	bounds['tau_m'] = [30., 40.]
	bounds['k_FR'] = [35., 45.]
	bounds['tau_FR'] = [0.45, 0.55]

	return bounds

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
	params['Nn'] = 5.0
	params['alpha_m'] = 2.0
	params['m_0'] = 1.5
	params['a_0'] = 0.33
	params['tau_m'] = 35.0
	params['k_FR'] = 40.
	params['tau_FR'] = 0.5
	
	return params
	
	
def bounds_Tar_1():
	
	state_bounds = [[1.0, 2.0], [0, 20]]

	param_bounds = OrderedDict()
	param_bounds['K_off_a'] = [0.01, 0.05]
	param_bounds['K_on_a'] = [0.2, 0.7]
	param_bounds['Nn'] = [4, 5]
	param_bounds['alpha_m'] = [0.5, 10]
	param_bounds['m_0'] = [0.5, 0.5]
	param_bounds['a_0'] = [1e-3, 1e0]
	param_bounds['tau_m'] = [1, 200]
	param_bounds['k_FR'] = [1, 200]
	param_bounds['tau_FR'] = [0.01, 5]

	bounds = dict()
	bounds['states'] = state_bounds
	bounds['params'] = param_bounds

	return bounds

def bounds_Tar_2():

	state_bounds = [[1.0, 2.0], [0, 20]]

	param_bounds = OrderedDict()
	param_bounds['K_off_a'] = [0.018, 0.022]
	param_bounds['K_on_a'] = [0.45, 0.55]
	param_bounds['Nn'] = [4.5, 5.5]
	param_bounds['alpha_m'] = [1.8, 2.1]
	param_bounds['m_0'] = [1.4, 1.6]
	param_bounds['a_0'] = [0.31, 0.35]
	param_bounds['tau_m'] = [30., 40.]
	param_bounds['k_FR'] = [35., 45.]
	param_bounds['tau_FR'] = [0.45, 0.55]

	bounds = dict()
	bounds['states'] = state_bounds
	bounds['params'] = param_bounds

	return bounds

def bounds_Tar_3():

    state_bounds = [[0.0, 5.0], [-20, 20]]

    param_bounds = OrderedDict()
    param_bounds['K_off_a'] = [0.01, 0.05]
    param_bounds['K_on_a'] = [0.5, 5.0]
    param_bounds['Nn'] = [3., 8.]
    param_bounds['alpha_m'] = [0.1, 10.0]
    param_bounds['m_0'] = [0.5, 5.0]
    param_bounds['a_0'] = [0.0, 1.0]
    param_bounds['tau_m'] = [1., 500.]
    param_bounds['k_FR'] = [0., 100.]
    param_bounds['tau_FR'] = [0, 100.]

    bounds = dict()
    bounds['states'] = state_bounds
    bounds['params'] = param_bounds

    return bounds

def bounds_Tar_4():

    state_bounds = [[0.0, 5.0], [-20, 20]]

    param_bounds = OrderedDict()
    param_bounds['K_off_a'] = [0.01, 0.02]
    param_bounds['K_on_a'] = [0.4, 0.6]
    param_bounds['Nn'] = [5., 5.]
    param_bounds['alpha_m'] = [0.1, 10.0]
    param_bounds['m_0'] = [0.5, 5.0]
    param_bounds['a_0'] = [0.3, 0.4]
    param_bounds['tau_m'] = [1., 100.]
    param_bounds['k_FR'] = [0., 100.]
    param_bounds['tau_FR'] = [0, 100.]

    bounds = dict()
    bounds['states'] = state_bounds
    bounds['params'] = param_bounds

    return bounds

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
import matplotlib.pyplot as plt
from single_cell_FRET import single_cell_FRET
from scipy.integrate import odeint
from params_bounds import params_Tar_1
from save_data import save_twin_data

FRET_noise = 2.0

a = single_cell_FRET()
a.import_signal_data()

params_dict = params_Tar_1()
params = []
for iP, val in enumerate(params_dict.values()):
	exec('params.append(%s)' % val)

x0 = sp.array([1.27, 7.0])
Tt = a.Tt[a.signal_bounds_lo: a.signal_bounds_hi]
states = odeint(a.df_data_generation, x0, Tt, args=(params, ))
stimuli = a.signal_vector[a.signal_bounds_lo: a.signal_bounds_hi]

save_twin_data(Tt, states, stimuli, measured_vars_and_noise=[[1, FRET_noise]])
								

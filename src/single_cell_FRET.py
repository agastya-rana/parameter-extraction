"""
Class for generating twin data and running variational annealing 
on single cell FRET data.

Created by Nirag Kadakia at 23:00 10-16-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp
import sys
import random
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from load_data import load_preliminary_FRET, load_protocol
from models import MWC_Tar

class single_cell_FRET():
	"""
	Class to hold single cell FRET functions and modules.
	"""
	
	def __init__(self):
		self.dt = 0.1
		self.nT = 500
		self.Tt = sp.arange(0, self.dt*self.nT, self.dt)
		self.model = MWC_Tar
		self.true_states = None
		self.true_params = None
		self.signal_vector = None
		self.x_integrate_init = None
		
	def import_signal_data(self, data_set=1, cell=12, nSkip=50, yscale=1e-3):
		self.signal_vector = load_preliminary_FRET(data_set=self.data_set, \
													cell=self.cell)['signal']
		self.signal_vector = self.signal_vector*yscale
		self.signal_vector = self.signal_vector[nSkip:] 

	def set_step_signal(self, density=20, seed=2, yvals=[.085, 0.1, 0.115]):
		switch_points = range(self.nT)
		sp.random.seed(seed)
		sp.random.shuffle(switch_points)
		switch_points = sp.sort(switch_points[:density-1])
		
		self.signal_vector = sp.zeros(self.nT)
		
		sp.random.seed(seed)
		self.signal_vector[:switch_points[0]] = random.choice(yvals)
		for iP in range(density-2):
			self.signal_vector[switch_points[iP]: switch_points[iP+1]] = \
								sp.random.choice(yvals)
		self.signal_vector[switch_points[density-2]:] = sp.random.choice(yvals)
		
	def signal(self, t):
		assert self.signal_vector is not None, 'Must set stimulus vector' \
												'before setting stimulus' \
												'function'
		
		return interp1d(self.Tt, self.signal_vector, 
						fill_value='extrapolate')(t)
		
	def df_data_generation(self, x, t, p):
		self.stim = self.signal(t)
		return self.model(t, x, (p, self.stim))
	
	def df_estimation(self, t, x, (p, stim)):
		return self.model(t, x, (p, stim))

	def df_integrate(self):
		assert self.x_integrate_init is not None, 'Set initial value'
		assert self.true_params is not None, 'Set true parameters'

		self.true_states = odeint(self.df_data_generation, 
								self.x_integrate_init, 
								self.Tt, args=(self.true_params, ))

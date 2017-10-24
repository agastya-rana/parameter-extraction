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
from utils import smooth
from params_bounds import *


class single_cell_FRET():
	"""
	Class to hold single cell FRET functions and modules.
	"""
	
	def __init__(self):

		self.nD = 2
		self.dt = 0.1
		self.nT = 500
		self.Tt = None	
	
		# Variables for integrating model / twin data generation
		self.model = MWC_Tar
		self.true_states = None
		self.true_params = None
		self.signal_vector = None
		self.x_integrate_init = None

		# Variables for state and parameter estimation
		self.Lidx = [1]
		self.param_bounds = None	
		self.state_bounds = None
		self.bounds = None
		self.nPest = None
		self.Pidx = None
		self.init_seed = 0
		self.x_init = None
		self.p_init = None

		# Variables for annealing
		self.alpha = 2.0
		self.beta_array = sp.linspace(0, 60, 61)
		self.Rf0 = 1e-6
		self.Rm = 1.0

	def set_Tt(self):	
		self.Tt = sp.arange(0, self.dt*self.nT, self.dt)
		
	def import_signal_data(self, data_set=1, cell=12, nSkip=50, yscale=1e-3):
		self.signal_vector = load_preliminary_FRET(data_set=self.data_set, \
													cell=self.cell)['signal']
		self.signal_vector = self.signal_vector*yscale
		self.signal_vector = self.signal_vector[nSkip:] 

	def set_step_signal(self, density=30, seed=20, yvals=[.085, 0.1, 0.115]):
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
		
	def smooth_step_signal(self, window_len=5):
		assert self.signal_vector is not None, \
			'Must set stimulus vector before smoothing'

		self.signal_vector = smooth(self.signal_vector, window_len=window_len)
		self.signal_vector = self.signal_vector[:self.nT]

	def signal(self, t):
		assert self.signal_vector is not None, \
			'Must set stimulus vector before setting stimulus function'
		assert self.Tt is not None, \
			'Must first set the time vector with set_Tt()'

		return interp1d(self.Tt, self.signal_vector, 
						fill_value='extrapolate')(t)

	def set_true_params(self, params_dict=params_Tar_1):
		self.true_params = []
		for iP, val in enumerate(params_dict().values()):
			exec('self.true_params.append(%s)' % val)

	def set_param_bounds(self, bounds_dict=bounds_Tar_1):
		param_bounds_dict = bounds_dict()['params']
		self.param_bounds = []
		
		for iP, val in enumerate(param_bounds_dict.values()):
			exec('self.param_bounds.append(%s)' % val)
		
		self.param_bounds = sp.array(self.param_bounds)
		self.nPest = len(self.param_bounds)
		self.Pidx = range(self.nPest)
	
	def set_state_bounds(self, bounds_dict=bounds_Tar_1):
		self.state_bounds = bounds_dict()['states']
		assert len(self.state_bounds) == self.nD, \
			'State bounds list must be state dimension length %s' % nD
		
		self.state_bounds = sp.array(self.state_bounds)

	def set_bounds(self):
		assert self.state_bounds is not None, \
			'First set state bounds through set_state_bounds(bounds_dict=...)'
		assert self.param_bounds is not None, \
			'First set param bounds through set_param_bounds(bounds_dict=...)'
		
		self.bounds = sp.vstack((self.state_bounds, self.param_bounds))

	def initial_estimate(self):
		assert self.bounds is not None, \
			'First bounds must be set through set_state_bounds(bounds_dict=' \
				'...), set_param_bounds(bounds_dict=...), and then ' \
				'set_bounds()'

		print 'Initialializing estimate with seed %s' % self.init_seed
		
		self.x_init = sp.zeros((self.nT, self.nD))
		self.p_init = sp.zeros(self.nPest)
		sp.random.seed(self.init_seed)

		for iD in range(self.nD):
			self.x_init[:, iD] = sp.random.uniform(
									self.state_bounds[iD][0], 
									self.state_bounds[iD][1], self.nT)
		
		for iP in range(self.nPest):
			self.p_init[iP] = sp.random.uniform(
	                                self.param_bounds[iD][0],
    	                            self.param_bounds[iD][1])

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

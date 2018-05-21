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
import scipy.linalg as LA
from load_data import load_stim_file, load_meas_file
import models
from utils import smooth_vec

INT_PARAMS = ['nT', 'nD', 'step_stim_density', 'step_stim_seed']
LIST_PARAMS = ['L_idxs', 'x0', 'step_stim_vals', 'P_idxs', 'beta_array']
MODEL_PARAMS = ['model']
STR_PARAMS = ['stim_file', 'stim_type', 'stim_smooth_type', 'params_set', 
				'bounds_set', 'meas_file']

class single_cell_FRET():
	"""
	Class to hold single cell FRET functions and modules.
	"""
	
	def __init__(self, **kwargs):

		self.nD = 2
		self.dt = 0.1
		self.nT = 500
		self.Tt = None
		self.L_idxs = [1]
		
		# Variables for generating fake stimulus consisting of steps
		self.stim = None
		self.stim_file = None
		self.stim_type = 'step'
		self.stim_smooth_dt = None
		self.stim_smooth_type = 'gaussian'
		self.step_stim_seed = 1
		self.step_stim_density = 30
		self.step_stim_vals = [0.085, 0.1, 0.115]
		
		# Variables for generating/loading measured data
		self.meas_data = None
		self.meas_file = None
		self.meas_data_seed = 0
		self.meas_noise = 1.0
		
		# Variables for integrating model/twin data generation
		self.model = models.MWC_Tar
		self.params_set = 'Tar_1'
		self.bounds_set = 'Tar_1'
		self.true_states = None
		self.x0 = [2.3, 7.0]
		
		# Variables for optimization of single annealing step
		self.nP = 9
		self.param_bounds = None	
		self.state_bounds = None
		self.bounds = None
		self.num_param_est = None
		self.param_est_idxs = None
		self.init_seed = 0
		self.x_init = None
		self.p_init = None

		# Variables for annealing run
		self.alpha = 2.0
		self.beta_array = range(0, 61)
		self.Rf0 = 1e-6

		# Variables for linear kernel estimation
		self.kernel_length = 50
		self.regularization = 1.0
		self.kernel_tikhonov_matrix = None
		self.kernel_inverse_hessian = None
		self.kernel_Aa = None
		self.kernel_estimator_nT = self.nT - self.kernel_length

				
		# Overwrite variables with passed arguments	
		for key in kwargs:
			
			assert hasattr(self, '%s' % key), "%s not a single_cell_FRET "\
				"attribute. Double-check or add to __init__" % key
			
			val = kwargs[key]
			
			if key in INT_PARAMS:
				exec ('self.%s = int(val)' % key)
			elif key in STR_PARAMS:
				exec ('self.%s = str(val)' % key)
			elif key in MODEL_PARAMS:
				assert hasattr(models, '%s' % val), 'Model class "%s" not in '\
					'models module' % val
				exec ('self.%s = models.%s' % (key, val))
			elif key in LIST_PARAMS:
				try:
					exec('len(%s)' % val)
				except:
					print 'The value of %s (%s) is not a list' % (key, val)
					quit()
				exec ('self.%s = %s' % (key, val))
			else:
				exec ('self.%s = float(val)' % key)
				
				
				
	#############################################
	##### 		Stimulus functions			#####
	#############################################

	

	def set_stim(self):
		"""
		Set the stimulus vector, either from file or generated from a function.
		"""
		
		if self.stim_file is not None:
			self.import_stim_data()
			print 'Stimulus data imported from %s.txt.' % self.stim_file
		else:
			if self.stim_type == 'step':
				self.set_step_stim()
			else:
				print 'Stimulus type stim_type=%s unknown' % self.stim_type	
				quit()
	
		if self.stim_smooth_dt is not None:
			window_len = int(1.*self.stim_smooth_dt/self.dt)
			self.stim = smooth_vec(self.stim, window_len, self.stim_smooth_type)
		
	def import_stim_data(self):
		"""
		Import stimulus from file.
		"""
		
		Tt_stim = load_stim_file(self.stim_file)
		
		assert Tt_stim.shape[0] == self.nT, "Loaded stimulus file has %s "\
			"timepoints, but nT is set to %s" % (Tt_stim.shape[0], self.nT)
		assert Tt_stim[1, 0] - Tt_stim[0, 0] == self.dt, "Loaded stimulus "\
			"file has timestep %s, but self.dt is %s" % (Tt_stim[1, 0] \
			- Tt_stim[0, 0], self.dt)
			
		self.Tt = Tt_stim[:, 0]
		self.stim = Tt_stim[:, 1]

	def set_step_stim(self):
		"""
		Generate a random step stimulus with given density of steps 
		and given discrete stimulus values.
		"""
	
		assert self.step_stim_density < self.nT, "step_stim_density must be "\
			"less than number of time points, but nT = %s, density = %s" \
			% (self.nT, self.step_stim_density)
		
		self.stim = sp.zeros(self.nT)
		
		# Get points at which to switch the step stimulus
		sp.random.seed(self.step_stim_seed)
		switch_pts = sp.random.choice(self.nT, self.step_stim_density)
		switch_pts = sp.sort(switch_pts)
		
		# Set values in each inter-switch interval from step_stim_vals array
		sp.random.seed(self.step_stim_seed)
		for iT in range(self.step_stim_density - 1):
			stim_val = sp.random.choice(self.step_stim_vals)
			self.stim[switch_pts[iT]: switch_pts[iT + 1]] = stim_val
		
		# Fill in ends
		edge_vals = sp.random.choice(self.step_stim_vals, 2)
		self.stim[:switch_pts[0]] = edge_vals[0]
		self.stim[switch_pts[self.step_stim_density - 1]:] = edge_vals[1]
		self.Tt = sp.arange(0, self.dt*self.nT, self.dt)
		
	def eval_stim(self, t):
		"""
		Evaluate the stimulus at arbitrary times by interpolated self.stim
		"""
		
		assert self.stim is not None, 'Set stim before smoothing w/ set_stim()'
		return interp1d(self.Tt, self.stim, fill_value='extrapolate')(t)

		
		
	#############################################
	#####          Measured Data            #####
	#############################################
	
	
	
	def set_meas_data(self):
		"""
		Set the meas data from file (if meas_file set) or gen from true states
		"""
		
		if self.meas_file is not None:
			self.import_meas_data()
			print 'Measured data imported from %s.txt.' % self.meas_file
		else:
			assert self.true_states is not None, "Since no measurement file "\
				"is specified, twin data will be generated. But before calling "\
				"set_meas_data(), you must first generate true data with "\
				"gen_true_states."
			
			self.meas_data = sp.zeros((self.nT, len(self.L_idxs)))
			sp.random.seed(self.meas_data_seed)
			self.meas_data = self.true_states[:, self.L_idxs] + \
				sp.random.normal(0, self.meas_noise, size=self.meas_data.shape)
			
	def import_meas_data(self):
		"""
		Import measured data from file
		"""
		
		Tt_meas_data = load_meas_file(self.stim_file)
		self.meas_data = Tt_meas_data[:, 1:]
		
		assert (sp.all(Tt_meas_data[:, 0] == self.Tt)), "Tt vector in "\
				"measured data file must be same as Tt"
		assert self.meas_data.shape[1] == len(self.L_idxs), "Dimension of "\
			"imported measurement vectors must be same as length of L_idxs"

			
		
	#############################################
	#####			VA parameters			#####
	#############################################

	
	
	def set_init_est(self):
		
		print 'Initializing estimate with seed %s' % self.init_seed
		
		assert (self.nD == self.model().nD), 'self.nD != %s.nD' % self.model
		assert (self.nP == self.model().nP), 'self.nP != %s.nP' % self.model
		
		self.state_bounds = self.model().bounds[self.bounds_set]['states']
		self.param_bounds = self.model().bounds[self.bounds_set]['params']
		self.bounds = sp.vstack((self.state_bounds, self.param_bounds))
		self.x_init = sp.zeros((self.nT, self.nD))
		self.p_init = sp.zeros(self.nP)
		
		sp.random.seed(self.init_seed)
		for iD in range(self.nD):
			self.x_init[:, iD] = sp.random.uniform(self.state_bounds[iD][0], 
									self.state_bounds[iD][1], self.nT)
		for iP in range(self.nP):
			self.p_init[iP] = sp.random.uniform(self.param_bounds[iD][0],
    	                            self.param_bounds[iD][1])

	def df_estimation(self, t, x, (p, stim)):
		"""
		Function to return value of vector field in format used for varanneal
		"""
		
		return self.model().df(t, x, (p, stim))

	
	
	#############################################
	#####	Twin data generation functions	#####
	#############################################

	
	
	def df_data_generation(self, x, t, p):
		"""
		Function to return value of vector field in format used for sp.odeint
		"""
		
		return self.model().df(t, x, (p, self.eval_stim(t)))
	
	def gen_true_states(self):
		"""
		Forward integrate the model given true parameters and x0
		"""
		
		self.true_states = odeint(self.df_data_generation, self.x0, self.Tt, 
									args=(self.model().params[self.params_set], ))
		


	#############################################
	#####		Kernel estimation			#####
	#############################################


	def set_kernel_estimation_Aa(self, stimulus=None):
		assert 0 <  self.kernel_length < self.nT, \
			'Kernel length must be nonzero and shorter than recorded time length'

		if stimulus is None:
			assert self.stim is not None, \
				'Signal vector must first be imported or set by' \
					' set_step_stim() or import_stim_data().' \
					' Or pass directly to set_kernel_estimation_Aa(stimulus=)'
		else:
			assert len(stimulus) == self.nT, \
				'User-defined stimulus in set_kernel_estimation_Aa(stimulus)' \
					' must be length %s' % self.nT
			self.stim = stimulus

		mean_subtracted_stimulus = self.stim - sp.average(self.stim)

		self.kernel_estimator_nT = self.nT - self.kernel_length
		self.kernel_Aa = sp.zeros((self.kernel_estimator_nT, self.kernel_length))
		for row in range(self.kernel_estimator_nT):
			self.kernel_Aa[row, :] = \
				mean_subtracted_stimulus[row: row + self.kernel_length]

	def set_kernel_estimation_regularization(self):
		self.kernel_tikhonov_matrix = sp.eye(self.kernel_length)*self.regularization

	def set_kernel_estimation_inverse_hessian(self):
		assert self.kernel_Aa is not None, \
			'First set kernel_Aa with set_kernel_estimation_Aa'

		self.kernel_inverse_hessian = \
			LA.inv(sp.dot(self.kernel_Aa.T, self.kernel_Aa) \
							+ self.kernel_tikhonov_matrix)
	
	def kernel_calculation(self, response_vector):
		assert self.kernel_inverse_hessian is not None, \
			'First set inverse hessian for the estimation with' \
				' set_inverse_hessian()'
		assert len(sp.shape(response_vector)) == 1 and \
				len(response_vector) == self.kernel_estimator_nT, \
			'response_vector should be numpy array with length equal to' \
				' stimulus data minus kernel length'

		mean_subtracted_response_vector = response_vector - sp.average(response_vector)

		estimated_kernel = sp.dot(sp.dot(self.kernel_inverse_hessian, 
								self.kernel_Aa.T), mean_subtracted_response_vector)
	
		return estimated_kernel

	def convolve_stimulus_kernel(self, kernel, stimulus=None):		
		if stimulus is None:
			assert self.stim is not None, \
				'Signal vector must first be imported or set by' \
					' set_step_stim() or import_stim_data().' \
					' Or pass directly to set_kernel_estimation_Aa(stimulus=)'
		else:
			assert len(stimulus) == self.nT, \
				'User-defined stimulus in set_kernel_estimation_Aa(stimulus)' \
				' must be length %s' % self.nT
			self.stim = stimulus

		mean_subtracted_stimulus = self.stim - sp.average(self.stim) 
		self.kernel_length = len(kernel)
		mean_subtracted_response_vector = sp.zeros(self.nT)

		for iT in range(self.nT):
			if iT >= self.kernel_length:
				mean_subtracted_response_vector[iT] = \
					sp.sum(mean_subtracted_stimulus[iT - self.kernel_length: iT]*kernel)
			else:
				mean_subtracted_response_vector[iT] = \
					sp.sum(mean_subtracted_stimulus[:iT]*kernel[self.kernel_length - iT:])
		
		return mean_subtracted_response_vector

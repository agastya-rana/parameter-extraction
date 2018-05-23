"""
MWC models and state bounds for FRET data assimilation.

Created by Nirag Kadakia at 08:40 05-20-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import scipy as sp
from collections import OrderedDict


class generic_model_class():
	
	def __init__(self, **kwargs):

		# State and parameter dimensions
		self.nD = 3
		self.nP = 2
	
		# List of state and parameter names
		self.state_names = ['state_1', 'state_2', 'state_3']
		self.param_names = ['param_1', 'param_2']
		
		# Dictionary of true parameter sets. The values of each dictionary is a 
		# list of length self.nP. The values in these lists are the parameter
		# values for the ith parameter. If true parameters are not known 
		# (i.e. twin data is not being generated), these can be omitted.
		self.params = dict()
		
		self.params['param_set_1'] = [0.02, 0.01]
		self.params['param_set_2'] = [0.92, 2.13]
		self.params['param_set_3'] = [3.03, 0.08]
		
		# Dictionary holding two dictionaries, one for parameter 
		# bounds one for state bounds. The values of these two dictionaries 
		# is a list of length self.nD and self.nP, respectively. Each element 
		# of these lists is in turn a 2-element list for the lower and upper
		# bounds of the ith state and ith parameter, respectively.
		self.bounds = dict()
		
		self.bounds['bounds_set_1'] = dict()
		self.bounds['bounds_set_1']['states'] = [[1.0, 5.0], [10, 100], [5, 17]]
		self.bounds['bounds_set_1']['parameters'] = [[0.001, 1.0], [0, 500]]
	
		self.bounds['bounds_set_2'] = dict()
		self.bounds['bounds_set_2']['states'] = [[1.0, 5.0], [2, 200], [0, 50]]
		self.bounds['bounds_set_2']['parameters'] = [[0, 5.0], [0, 100]]
	
	def df(self, t, x, (p, stim)):
		"""
		The vector field function. 
		
		Args:
			t: float; time at which to evaluate non-autonomous vector field.
			x: numpy array of arbitrary shape, provided axis -1 
						has length self.nD. This allows vectorized 
						evaluation.
			p: list of length self.nP giving float-values of model parameters.
			stim: float; value of stimulus at time t.

		Returns:
			df_vec: numpy array of shape x; vector field.
		"""

		# Unpack states and parameters as shown below.
		x1 = x[...,0]
		x2 = x[...,1]
		x3 = x[...,2]
		
		p1, p2, p3 = p
		
		df_vec = sp.empty_like(x)	

		df_vec[..., 0]  = -x1**2.0 + 3*(p1 + p2)
		df_vec[..., 1]  = -x1 + x3*p3**2.0
		df_vec[..., 2]  = -x3/p2

		return df_vec
	
	
class MWC_Tar():
	""" 
	2-variable MWC FRET model class; FRET index is dynamic. 
	Two dynamical variables are methylation state and FRET index, which 
	relaxes to a scaled activity level.
	"""
	
	def __init__(self, **kwargs):

		
		self.nD = 2
		self.nP = 9
		self.state_names = ['methyl', 'FRET index']
		self.param_names = ['K_off_a', 
							'K_on_a', 
							'Nn', 
							'alpha_m', 
							'm_0',
							'a_0', 
							'tau_m', 
							'k_FR', 
							'tau_FR']
		
		
		# True parameter dictionaries
		self.params = dict()
		self.params['Tar_1'] = [0.02, 
								0.5, 
								5.0, 
								2.0, 
								1.5, 
								0.33, 
								35.0, 
								40.0, 
								0.5]
		
		# Bounds dictionaries
		self.bounds = dict()
		self.bounds['Tar_1a'] = dict()
		self.bounds['Tar_1a']['states'] = 	[[1.0, 2.0], [0, 20]]
		self.bounds['Tar_1a']['params'] = 	[[0.01, 0.05],
											[0.2, 0.7],
											[4, 6],
											[0.1, 10],
											[0.1, 10],
											[1e-3, 1e0],
											[1, 200],
											[1, 200],
											[0.01, 5]]
			
		self.bounds['Tar_1b'] = dict()
		self.bounds['Tar_1b']['states'] = 	[[1.0, 2.0], [0, 20]]
		self.bounds['Tar_1b']['params'] = 	[[0.018, 0.022],
											[0.45, 0.55],
											[4.5, 5.5],
											[1.8, 2.1],
											[1.4, 1.6],
											[0.31, 0.35],
											[30., 40.],
											[35., 45.],
											[0.45, 0.55]]
					
	def df(self, t, x, (p, stim)):
		"""
		Taken from Clausznitzer,...,Sourjik, Endres 2014 PLoS Comp Bio.
		Only Tar receptor, Tsr not included.  
		"""

		Mm = x[...,0]
		FR = x[...,1]
		K_off_a, K_on_a, Nn, \
			alpha_m, m_0,  \
			a_0, tau_m,  \
			k_FR, tau_FR = p

		df_vec = sp.empty_like(x)	

		f_c = sp.log((1. + stim/K_off_a)/(1. + stim/K_on_a))
		f_m = alpha_m*(m_0 - Mm)
		Ee = Nn*(f_m + f_c)
		Aa = (1. + sp.exp(Ee))**-1.0

		df_vec[..., 0]  = (a_0 - Aa)/tau_m
		df_vec[..., 1]  = k_FR*Aa - FR/tau_FR

		return df_vec
		
		
class MWC_MM_2_var():
	""" 
	2-variable MWC with Michaelis Minton methylation dynamics. 
	The dynamical variables are methylation state and FRET index. 
	
	In general, we assume K_I, m_0, alpha_m, K_R and K_B are fixed. So 
	one may hold these parameter bounds as tight around some presumed value.
	"""
	
	def __init__(self, **kwargs):

		self.nD = 2
		self.nP = 9
		self.state_names = ['methyl', 'FRET index']
		self.param_names = ['K_I', 
							'm_0', 
							'alpha_m', 
							'K_R', 
							'K_B', 
							'Nn', 
							'V_R', 
							'V_B',
							'FR_scale']
		
		# True parameter dictionaries
		self.params = dict()
		self.params['1'] = [18., 	# K_I binding constant
							0.5, 	# m_0 bkgrnd methyl level
							2.0, 	# alpha_m 
							0.32, 	# K_R
							0.30,	# K_B 
							5.0, 	# N cluster size
							0.015, 	# V_R
							0.012,	# V_B
							50.0]	# a-->FRET scalar
		
		# Bounds dictionaries
		self.bounds = dict()
		self.bounds['VR_VB'] = dict()
		self.bounds['VR_VB']['states'] = [[0.0, 10.0], [-100, 100]]
		self.bounds['VR_VB']['params'] = [[18, 18],		# K_I binding constant
										[0.5, 0.5],		# m_0 bkg methyl level
										[2., 2.],		# alpha_m 
										[0.32, 0.32],	# K_R
										[0.30, 0.30],	# K_B 
										[5., 5.],		# N cluster size
										[1e-3, 1],		# V_R
										[1e-3, 1], 		# V_B
										[50., 50.]]		# a-->FRET scalar
		self.bounds['1a'] = dict()
		self.bounds['1a']['states'] = [[0.0, 10.0], [-100, 100]]
		self.bounds['1a']['params'] = [[1, 50],			# K_I binding constant
										[0, 10],		# m_0 bkg methyl level
										[1, 10],		# alpha_m 
										[0.32, 0.32],	# K_R
										[0.30, 0.30],	# K_B 
										[1, 10],		# N cluster size
										[1e-3, 1],		# V_R
										[1e-3, 1], 		# V_B
										[0, 100]]		# a-->FRET scalar
		self.bounds['1b'] = dict()
		self.bounds['1b']['states'] = [[0.0, 10.0], [-100, 100]]
		self.bounds['1b']['params'] = [[15, 25],		# K_I binding constant
										[0.5, 0.5],		# m_0 bkg methyl level
										[2.0, 2.0],		# alpha_m 
										[0.32, 0.32],	# K_R
										[0.30, 0.30],	# K_B 
										[1, 10],		# N cluster size
										[1e-3, 1],		# V_R
										[1e-3, 1], 		# V_B
										[1, 100]]		# a-->FRET scalar
		self.bounds['1c'] = dict()
		self.bounds['1c']['states'] = [[0.0, 10.0], [-100, 100]]
		self.bounds['1c']['params'] = [[18, 18],		# K_I binding constant
										[0.5, 0.5],		# m_0 bkg methyl level
										[2.0, 2.0],		# alpha_m 
										[0.32, 0.32],	# K_R
										[0.30, 0.30],	# K_B 
										[1, 50],		# N cluster size
										[1e-3, 1],		# V_R
										[1e-3, 1], 		# V_B
										[1, 100]]		# a-->FRET scalar
										
	def df(self, t, x, (p, stim)):
		
		Mm = x[...,0]
		FR_idx = x[...,1]
		K_I, m_0, alpha_m, K_R, K_B, Nn, V_R, V_B, FR_scale = p
		
		df_vec = sp.empty_like(x)	
		
		f_c = sp.log(1. + stim/K_I)
		f_m = alpha_m*(m_0 - Mm)
		Ee = Nn*(f_m + f_c)
		Aa = 1/(1. + sp.exp(Ee))

		df_vec[..., 0] = V_R*(1 - Aa)/(K_R + (1 - Aa)) \
						- V_B*Aa**2/(K_B + Aa)
		df_vec[..., 1]  = FR_scale*Aa - FR_idx/0.5
		
		return df_vec
		

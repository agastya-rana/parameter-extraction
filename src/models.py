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


class MWC_Tar():
	""" 
	MWC FRET model class
	"""
	
	def __init__(self, **kwargs):

		self.param_names = ['K_off_a', 'K_on_a', 'Nn', 'alpha_m', 'm_0',
							'a_0', 'tau_m', 'k_FR', 'tau_FR']
		
		# Set state and parameter dimensions
		self.nD = 2
		self.nP = 9
		
		# Set parameter dictionaries
		self.params = dict()
		self.params['Tar_1'] = [0.02, 0.5, 5., 2., 1.5, 0.33, 35., 40., 0.5]
		
		# Set bounds dictionaries
		self.bounds = dict()
		self.bounds['Tar_1a'] = dict()
		self.bounds['Tar_1a']['states'] = [[1.0, 2.0], [0, 20]]
		self.bounds['Tar_1a']['params'] = [[0.01, 0.05],
										[0.2, 0.7],
										[4, 5],
										[0.5, 10],
										[0.5, 0.5],
										[1e-3, 1e0],
										[1, 200],
										[1, 200],
										[0.01, 5]]
		
		self.bounds['Tar_1b'] = dict()
		self.bounds['Tar_1b']['states'] = [[1.0, 2.0], [0, 20]]
		self.bounds['Tar_1b']['params'] = [[0.018, 0.022],
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
		The vector field, must be called 'df'.
		Taken from Clausznitzer,...,Sourjik, Endres 2014 PLoS Comp Bio.
		Only Tar receptor, Tsr not included.  
		"""

		Mm = x[...,0]
		FR = x[...,1]
		
		df_vec = sp.empty_like(x)	

		K_off_a, K_on_a, Nn, \
			alpha_m, m_0,  \
			a_0, tau_m,  \
			k_FR, tau_FR = p

		f_c = sp.log((1. + stim/K_off_a)/(1. + stim/K_on_a))
		f_m = alpha_m*(m_0 - Mm)
		Ee = Nn*(f_m + f_c)
		Aa = (1. + sp.exp(Ee))**-1.0

		df_vec[..., 0]  = (a_0 - Aa)/tau_m
		df_vec[..., 1]  = k_FR*Aa - FR/tau_FR

		return df_vec
		
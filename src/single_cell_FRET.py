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
from scipy.interpolate import interp1d
from load_data import load_preliminary_FRET, load_protocol
from models import MWC_Tar

class single_cell_FRET():
	"""
	Class to hold single cell FRET functions and modules.
	"""
	
	def __init__(self):
		self.data_set = 1
		self.cell = 12
		
		self.data_dt = 0.5
		self.signal_bounds_lo = 50
		self.signal_bounds_hi = 600
		self.signal_convert_factor = 1e-3
	
		self.model = MWC_Tar
		
	def import_signal_data(self):
		self.signal_vector = load_preliminary_FRET(data_set=self.data_set, \
													cell=self.cell)['signal']
		self.signal_vector = self.signal_vector*self.signal_convert_factor
		self.Tt = sp.arange(0, len(self.signal_vector)*self.data_dt, 
								self.data_dt)
		
	def lorenz_signal_data(self, dilate=1.0):
		self.signal_vector = load_protocol(type='lorenz', params=[dilate])  

	def signal(self, t):
		return interp1d(self.Tt, self.signal_vector)(t)
		
	def df_data_generation(self, x, t, p):
		self.stim = self.signal(t)
		return self.model(t, x, (p, self.stim))
	
	def df_estimation(self, t, x, (p, stim)):
		return self.model(t, x, (p, stim))

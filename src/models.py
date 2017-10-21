"""
MWC models for bacterial chemotaxis.

Created by Nirag Kadakia at 08:40 10-16-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import scipy as sp

def MWC_Tar(t, x, (p, stim)):
	"""
	Taken from Clausznitzer,...,Sourjik, Endres 2014 PLoS Comp Bio.
	Only Tar receptor, Tsr not included.  
	"""

	if len(sp.shape(x)) == 2:
		Mm = x.T[0].T
		FR = x.T[1].T
		df = sp.zeros(x.shape, dtype='object')
	elif len(sp.shape(x)) == 1:
		Mm, FR = x
		df = sp.zeros(x.shape)
	
	K_off_a, K_on_a, Nn, \
		alpha_m, m_0,  \
		a_0, tau_m,  \
		k_FR, tau_FR = p

	f_c = sp.log((1. + stim/K_off_a)/(1. + stim/K_on_a))
	f_m = alpha_m*(m_0 - Mm)
	Ee = Nn*(f_m + f_c)
	Aa = (1. + sp.exp(Ee))**-1.0
	
	if len(sp.shape(x)) == 2:
		df[:, 0]  = (a_0 - Aa)/tau_m
		df[:, 1]  = k_FR*Aa - FR/tau_FR
	elif len(sp.shape(x)) == 1:
		df[0] = (a_0 - Aa)/tau_m
		df[1] = k_FR*Aa - FR/tau_FR

	return df


"""
Variational annealing of single cell FRET data. 

Created by Nirag Kadakia at 08:00 10-16-2017
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys, time
sys.path.append('../src')
sys.path.append('../../../../varanneal_NK/varanneal/varanneal')
sys.path.append('/home/fas/emonet/nk479/varanneal_NK/varanneal/varanneal')
import va_ode
import scipy as sp
#from varanneal import va_ode
from utils import get_flags
from single_cell_FRET import single_cell_FRET
from params_bounds import bounds_Tar_1, bounds_Tar_2
from load_data import load_VA_data
from save_data import save_estimates

data_flags = get_flags()

n_ID = data_flags[0]
data_dt = float(data_flags[1])
data_sigma = float(data_flags[2])

nD = 2
Lidx = [1]
Pidx = 'All'
param_bounds_dict = bounds_Tar_1()
state_bounds = [[0.0, 10.0], [3.0, 12.0]]

# Annealing parameters
alpha = 2.0
beta_array = sp.linspace(0, 50, 51)
RF0 = 1e-7
RM = 1.0 / data_sigma**2.0
cutoff = 1 #if nT is even-D

# Load twin data from file
data_dict = load_VA_data(data_flags=data_flags) 
measurements = data_dict['measurements'][:-cutoff, 1:]
stimuli = data_dict['stimuli'][:-cutoff]
Tt = data_dict['measurements'][:-cutoff, 0]
nT = len(Tt)
dt = Tt[1] - Tt[0]

# Set parameter bounds from file
param_bounds = []
for iP, val in enumerate(param_bounds_dict.values()):
	exec('param_bounds.append(%s)' % val)
nP = len(param_bounds)

# Combine bounds
state_bounds = sp.array(state_bounds)
full_bounds = sp.vstack((state_bounds, param_bounds))

# Set initial guesses
x_init = sp.zeros((nT, nD))
for iD in range(nD):
	x_init[:, iD] = sp.random.uniform(state_bounds[iD][0], state_bounds[iD][1], nT)
p_init = []
for iP in range(nP):
	p_init.append(sp.random.uniform(param_bounds[iP][0], param_bounds[iP][1]))
if Pidx == 'All':
	Pidx = sp.arange(nP)
p_init = sp.array(p_init)

# Initialize annealer and FRET class 
annealer = va_ode.Annealer()
scF = single_cell_FRET()
annealer.set_model(scF.df_estimation, nD)
annealer.set_data(measurements, stim=stimuli,  t=Tt)

# Estimate
BFGS_options = {'gtol':1.0e-8, 'ftol':1.0e-8, 'maxfun':1000000, 'maxiter':1000000}
tstart = time.time()
annealer.anneal(x_init, p_init, alpha, beta_array, RM, RF0, Lidx, Pidx, dt_model=None, 
               init_to_data=True, bounds=full_bounds,  disc='SimpsonHermite', 
               method='L-BFGS-B', opt_args=BFGS_options, adolcID=0)
print("\nADOL-C annealing completed in %f s."%(time.time() - tstart))

save_estimates(annealer, data_flags)

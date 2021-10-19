"""
MWC models and state bounds for FRET data assimilation.

Created by Nirag Kadakia at 08:40 05-20-2018
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import autograd.numpy as np
## TODO: change the input arguments to explicitly input parameter and stimulus
## TODO: need to write a kwargs part that allows users to define a model inherited from this class
## outside of this code... and somehow that model should be callable from the specs file.
from collections import OrderedDict


class Model():

    def __init__(self, nD=2, nP=3, **kwargs):
        # State and parameter dimensions
        self.nD = nD
        self.nP = nP

        # List of state and parameter names
        self.state_names = ['state_{}'.format(i) for i in range(nD)]
        self.param_names = ['param_{}'.format(i) for i in range(nP)]

        # Dictionary of true parameter sets. The values of each dictionary is a
        # list of length nP, storing the parameter values. If true parameters are not known
        # (i.e. twin data is not being generated), these can be omitted.
        self.params = dict()
        # self.params['init'] = [0.0, 0.0, 0.0]

        # Dictionary holding dictionaries of parameter and state bounds.
        # Within each subdictionary are lists of length nD and nP, corresponding
        # to the keys of 'states' and 'parameters' respectively. Each list stores
        # two-element lists of lower and upper bounds for each state/parameter.
        # Parameters can be fixed by setting their lower and upper bounds equal to each other.
        self.bounds = dict()

    ## self.bounds['init'] = dict()
    ## self.bounds['init']['states'] = [[0.,1.], [0., 100.]]
    ## self.bounds['init']['parameters'] = [[0.001, 1.0], [0, 500], [10., 10.]]

    def df(self, t, x, p, stim):
        """
		The dynamic model function, that returns the derivative of the state,
		used to predict the state at the next timestep.
		
		Args:
			t: float; time at which to evaluate non-autonomous vector field.
			x: numpy array of arbitrary shape, provided axis -1 
						has length self.nD. This allows vectorized 
						evaluation.
			p: list of length self.nP giving float-values of model parameters.
			stim: float; value of stimulus at time t.

		Returns:
			df_vec: numpy array of shape x; derivative df/dx.

		Note: to allow easy gradient evaluation, all functions defined in df must not
		assign elements to arrays individually.
		"""
        ## Note that x can be 2-D, such that the first axis is time (needed if the update rule is non-Markovian)
        x1 = x[..., 0]
        x2 = x[..., 1]
        p1, p2, p3 = p
        df_vec = np.array([x1 * p1, x2]).T
        return df_vec

class MWC_MM(Model):
    """
	MWC model for activity; Michaelis-Menten model for methylation/demethylation.
	Assume K_I, m_0, alpha_m, K_R, K_B are fixed; narrowly constrain these bounds.
	"""

    def __init__(self, **kwargs):
        self.nD = 2
        self.nP = 10
        self.state_names = ['methyl', 'FRET index']
        self.param_names = ['K_I',
                            'm_0',
                            'alpha_m',
                            'K_R',
                            'K_B',
                            'Nn',
                            'V_R',
                            'V_B',
                            'FR_scale',
                            'FR_shift']

        # True parameter dictionaries; TODO: data from where???
        self.params = dict()
        self.params['1'] = [18.,  # K_I binding constant
                            0.5,  # m_0 bkgrnd methyl level
                            2.0,  # alpha_m
                            0.32,  # K_R
                            0.30,  # K_B
                            5.0,  # N cluster size
                            0.015,  # V_R
                            0.012,  # V_B
                            50.0,  # a-->FRET scalar
                            0.0]  # FRET signal background shift

        # Bounds dictionaries
        self.bounds = dict()
        self.bounds['1a'] = dict()
        self.bounds['1a']['states'] = [[0.0, 5.0], [0, 100]]
        self.bounds['1a']['params'] = [[1, 100],  # K_I binding constant
                                       [0.5, 0.5],  # m_0 bkg methyl level
                                       [2.0, 2.0],  # alpha_m
                                       [0.0, 1.0],  # K_R
                                       [0.0, 1.0],  # K_B
                                       [0, 200],  # N cluster size
                                       [1e-3, 1],  # V_R
                                       [1e-3, 1],  # V_B
                                       [0, 100],  # a-->FRET scalar
                                       [-50, 50]]  # FRET y-shift

    def df(self, t, x, inputs):
        p, stim = inputs
        Mm = x[..., 0]
        FR_idx = x[..., 1]
        K_I, m_0, alpha_m, K_R, K_B, Nn, V_R, V_B, FR_scale, FR_shift = p

        f_c = np.log(1. + stim / K_I)
        f_m = alpha_m * (m_0 - Mm)
        Ee = Nn * (f_m + f_c)
        Aa = 1. / (1. + np.exp(Ee))

        df_vec = np.array([V_R * (1 - Aa) / (K_R + (1 - Aa)) \
                           - V_B * Aa ** 2 / (K_B + Aa), (FR_scale * Aa - FR_shift - FR_idx) / 0.5]).T
        return df_vec

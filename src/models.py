"""
MWC models and state bounds for FRET data assimilation.

Created by Nirag Kadakia and Agastya Rana, 10-21-2021.
"""

import autograd.numpy as np
## TODO: change the input arguments to explicitly input parameter and stimulus
## TODO: need to write a kwargs part that allows users to define a model inherited from this class
## outside of this code... and somehow that model should be callable from the specs file.


class Model():
    def __init__(self, nD=2, nP=3, **kwargs):
        # State and parameter dimensions
        self.nD = nD
        self.nP = nP
        self.x0 = [0,0] ## initial, equilibrium state of the system
        self.P_idxs = [-1] ## parameters that very
        self.L_idxs = [1] ## only a (not m) is observable in the [m,a] tuple

        # List of state and parameter names
        self.state_names = ['state_{}'.format(i) for i in range(nD)]
        self.param_names = ['param_{}'.format(i) for i in range(nP)]

        # Dictionary of true parameter sets. The values of each dictionary is a
        # list of length nP, storing the parameter values. If true parameters are not known
        # (i.e. twin data is not being generated), these can be omitted.
        self.params_set = []
        # self.params['init'] = [0.0, 0.0, 0.0]

        # Dictionary holding dictionaries of parameter and state bounds.
        # Within each subdictionary are lists of length nD and nP, corresponding
        # to the keys of 'states' and 'parameters' respectively. Each list stores
        # two-element lists of lower and upper bounds for each state/parameter.
        # Parameters can be fixed by setting their lower and upper bounds equal to each other.
        self.state_bounds = [[0.0, 4.0], [0, 1]]
        self.param_bounds = []

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
	Parameters here taken from Shimizu (2010), except for K_I from Clausnitzer 2014.
	"""

    def __init__(self, **kwargs):
        self.nD = 2
        self.nP = 9
        self.L_idxs = [1]
        self.P_idxs = [-3, -2, -1]
        self.state_names = ['methyl', 'FRET index']
        self.param_names = ['K_I', 'K_A', 'm_0', 'alpha_m', 'K_R', 'K_B', 'Nn', 'V_R', 'V_B']

        # True parameter dictionaries;
        self.params_set = [20., 3225., 0.5, 2.0, 0.32, 0.30, 6.0, 0.010, 0.013]
        a0 = 0.33 ## TODO: fill in equation for a where dm/dt=0
        self.x0 = [self.params_set[2]+0.83, a0]

        # Bounds dictionaries
        self.state_bounds = [[0.0, 4.0], [0, 1]]
        self.param_bounds = [[4, 8], [0.001, 0.1], [0.001, 0.1]] ## N, V_R, V_B

    def df(self, t, x, inputs):
        p, stim = inputs
        Mm = x[..., 0]
        FR_idx = x[..., 1]
        K_I, K_A, m_0, alpha_m, K_R, K_B = self.params_set[:6]
        Nn, V_R, V_B = p

        f_c = np.log((1. + stim / K_I)/(1. + stim / K_A))
        f_m = alpha_m * (m_0 - Mm)
        Ee = Nn * (f_m + f_c)
        Aa = 1. / (1. + np.exp(Ee))
        df_vec = np.array([V_R*(1 - Aa)/(K_R + (1 - Aa)) - V_B*Aa/(K_B + Aa), (Aa - FR_idx) / 0.5]).T
        ## TODO: implement explicit reliance on self.dt instead of 0.5
        return df_vec

class MWC_linear(Model):
    """
	MWC model for activity; Michaelis-Menten model for methylation/demethylation.
	Assume K_I, m_0, alpha_m, K_R, K_B are fixed; narrowly constrain these bounds.
	Parameters here taken from Shimizu (2010), except for K_I from Clausnitzer 2014.
	"""

    def __init__(self, **kwargs):
        self.nD = 2
        self.nP = 7
        self.L_idxs = [1]
        self.P_idxs = [-3, -2, -1]
        self.state_names = ['methyl', 'FRET index']
        self.param_names = ['K_I',
                            'K_A'
                            'm_0',
                            'alpha_m',
                            'Nn',
                            'a_ss',
                            'slope']
        # True parameter dictionaries;
        self.params_set = [20., 3225., 0.5, 2.0, 6.0, 0.33, -0.01]
        self.x0 = [self.params_set[2]+0.83, self.params_set[-2]] ## 0.83 correction needed for 100 uM bg

        # Bounds dictionaries
        self.state_bounds = [[0.0, 4.0], [0, 1]]
        self.param_bounds = [[4, 15], [0., 1.], [-0.5, 0.5]] ## N, a_ss, slope

    def df(self, t, x, inputs):
        p, stim = inputs
        Mm = x[..., 0]
        FR_idx = x[..., 1]
        K_I, K_A, m_0, alpha_m = self.params_set[:4]
        Nn, a_ss, slope = p

        f_c = np.log((1. + stim / K_I)/(1. + stim / K_A))
        f_m = alpha_m * (m_0 - Mm)
        Ee = Nn * (f_m + f_c)
        Aa = 1. / (1. + np.exp(Ee))
        dm = slope*(Aa-a_ss)
        df_vec = np.array([dm, (Aa - FR_idx) / 0.5]).T
        return df_vec


## TODO: add model class using sdeint - check why diff from odeint without noise, this way using both process and measurement noise
## Clausnitzer (2014) model has MWC with dm/dt = g_R(1-A) - g_B(A^3); K_i = 0.02 mM; K_a = 0.5 mM; N(c) = 17.5 + 3.35*c where c in mM; g_R = 0.0069; g_B = 0.11; f_m = 1- 0.5m
## They interpolated f_m from Endres RG, Oleksiuk O, Hansen CH, Meir Y, Sourjik V, et al. (2008)

## Shimizu (2010) use dose-response and show good fit for N = 6, K_I/K_A = 0.0062; use this to get alpha = 2, m = 0.5 (out of 4),
## To fit their kinked F(a), they use piecewise V_B(a); we instead will just focus on piece that corresponds to near a_0
## They have V_R = 0.010; V_B = 0.013; K_R = 0.32; K_B = 0.30.

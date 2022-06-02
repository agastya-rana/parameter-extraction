"""
Dynamical models and state bounds for FRET data assimilation.
Created by Nirag Kadakia and Agastya Rana, 10-21-2021.
"""

import autograd.numpy as np

class CellModel():
    nD = 2 ## dimensions of the model
    L_idxs = [1] ## state csomponent indices that are observable
    state_names = ['methyl', 'FRET index']
    dt = 0.5
    def __init__(self):
        self.nP = 0
        self.x0 = [0, 0]  ## initial, equilibrium state of the system
        self.constant_names = [] ## names of fixed constants
        self.param_names = [] ## names of variable parameters
        self.P_idxs = [-1]  ## parameter indices that very
        self.constant_set = [] ## True fixed parameters
        self.params_set = [] ## True variable parameter values
        self.state_bounds = [[0.0, 4.0], [0, 1]] ## bounds of state components
        self.param_bounds = [[4, 8], [0.001, 0.1], [0.001, 0.1]] ## [lower, upper] only for variable parameters

    def df(self, t, x, p, stim):
        """
        The dynamic model function that returns the derivative of the state, used to predict the state at the next timestep.
        Args:
            t: time, x: states,
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

class MWC_MM(CellModel):
    """
	MWC model for activity; Michaelis-Menten model for methylation/demethylation.
	Assume K_I, m_0, alpha_m, K_R, K_B are fixed; narrowly constrain these bounds.
	Parameters here taken from Shimizu (2010), except for K_I from Clausnitzer 2014.
	"""

    def __init__(self):
        super().__init__()
        self.constant_names = ['K_I', 'K_A', 'm_0', 'alpha_m', 'K_R', 'K_B']
        self.param_names = ['Nn', 'V_R', 'V_B']
        self.constant_set = [20., 3225., 0.5, 2.0, 0.32, 0.30]
        self.params_set = [6.0, 0.010, 0.013]
        self.x0 = [self.constant_set[2]+0.83, 0.33]
        self.param_bounds = [[4, 8], [0.001, 0.1], [0.001, 0.1]] ## N, V_R, V_B

    def df(self, t, x, inputs):
        p, stim = inputs
        Mm = x[..., 0]
        FR_idx = x[..., 1]
        K_I, K_A, m_0, alpha_m, K_R, K_B = self.constant_set
        Nn, V_R, V_B = p
        f_c = np.log((1. + stim / K_I)/(1. + stim / K_A))
        f_m = alpha_m * (m_0 - Mm)
        Ee = Nn * (f_m + f_c)
        Aa = 1. / (1. + np.exp(Ee))
        df_vec = np.array([V_R*(1 - Aa)/(K_R + (1 - Aa)) - V_B*Aa/(K_B + Aa), (Aa - FR_idx) / self.dt]).T
        return df_vec

class MWC_linear(CellModel):
    """
	MWC model for activity; linear model for methylation/demethylation.
	Assume K_I, m_0, alpha_m, K_R, K_B are fixed; narrowly constrain these bounds.
	Parameters here taken from Shimizu (2010), except for K_I from Clausnitzer 2014.
	"""

    def __init__(self):
        super().__init__()
        self.constant_names = ['K_I', 'K_A', 'm_0', 'alpha_m']
        self.param_names = ['Nn', 'a_ss', 'slope']
        self.constant_set = [20., 3225., 0.5, 2.0]
        self.params_set = [6.0, 0.33, -0.01]
        self.x0 = [self.constant_set[2]+0.83, self.params_set[1]] ## 0.83 correction needed for 100 uM bg
        self.param_bounds = [[4, 15], [0., 1.], [-0.5, 0.5]] ## N, a_ss, slope

    def df(self, t, x, inputs):
        p, stim = inputs
        Mm = x[..., 0]
        FR_idx = x[..., 1]
        K_I, K_A, m_0, alpha_m = self.constant_set
        Nn, a_ss, slope = p
        f_c = np.log((1. + stim / K_I)/(1. + stim / K_A))
        f_m = alpha_m * (m_0 - Mm)
        Ee = Nn * (f_m + f_c)
        Aa = 1. / (1. + np.exp(Ee))
        dm = slope*(Aa-a_ss)
        df_vec = np.array([dm, (Aa - FR_idx) / self.dt]).T
        return df_vec

class MWC_MM_Swayam(CellModel):
    """
    MWC model for activity; Michaelis-Menten model for methylation/demethylation.
    Assume K_I, m_0, alpha_m, K_R, K_B are fixed; narrowly constrain these bounds.
    Parameters here taken from Shimizu (2010), except for K_I from Clausnitzer 2014.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.constant_names = ['K_I', 'K_A', 'm_0', 'alpha_m', 'K_R', 'K_B']
        self.param_names = ['Nn', 'V_R', 'V_B']
        self.constant_set = [20., 3225., 0.5, 2.0, 1.0, 0.30]
        self.params_set = [6.0, 0.010, 0.013]
        self.x0 = [self.constant_set[2] + 0.83, 0.33]
        self.param_bounds = [[1, 30], [0.001, 0.1], [0.001, 0.1]]  ## N, V_R, V_B

    def df(self, t, x, inputs):
        p, stim = inputs
        Mm = x[..., 0]
        FR_idx = x[..., 1]
        K_I, K_A, m_0, alpha_m, K_R, K_B = self.constant_set
        Nn, V_R, V_B = p
        f_c = np.log((1. + stim / K_I) / (1. + stim / K_A))
        f_m = alpha_m * (m_0 - Mm)
        Ee = Nn * (f_m + f_c)
        Aa = 1. / (1. + np.exp(Ee))

        #df_vec = np.array([V_R * (1 - Aa) / (K_R + (1 - Aa)) - V_B * Aa / (K_B + Aa), (Aa - FR_idx) / self.dt]).T
        df_vec = np.array([np.exp(np.log(V_R)) * ((1 - Aa) / (K_R)) - np.exp(np.log(V_B)) * Aa / (K_B + Aa),
                           (Aa - FR_idx) / self.dt]).T
        return df_vec

## TODO: add model class using sdeint - check why diff from odeint without noise, this way using both process and measurement noise
## Clausnitzer (2014) model has MWC with dm/dt = g_R(1-A) - g_B(A^3); K_i = 0.02 mM; K_a = 0.5 mM; N(c) = 17.5 + 3.35*c where c in mM; g_R = 0.0069; g_B = 0.11; f_m = 1- 0.5m
## They interpolated f_m from Endres RG, Oleksiuk O, Hansen CH, Meir Y, Sourjik V, et al. (2008)

## Shimizu (2010) use dose-response and show good fit for N = 6, K_I/K_A = 0.0062; use this to get alpha = 2, m = 0.5 (out of 4),
## To fit their kinked F(a), they use piecewise V_B(a); we instead will just focus on piece that corresponds to near a_0
## They have V_R = 0.010; V_B = 0.013; K_R = 0.32; K_B = 0.30.
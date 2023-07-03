import autograd.numpy as np
from scipy.integrate import odeint
import sdeint
from scipy.interpolate import interp1d


class MWC_linear():
    """
	MWC model for activity; linear model for methylation/demethylation.
	Assume K_I, m_0, alpha_m, K_R, K_B are fixed; narrowly constrain these bounds.
	Parameters here taken from Shimizu (2010), except for K_I from Clausnitzer 2014.
	"""
    def __init__(self):
        self.nD = 2
        self.nP = 7
        self.L_idxs = [1]
        self.P_idxs = [-3, -2, -1]
        self.state_names = ['methyl', 'FRET index']
        self.param_names = ['K_I', 'K_A', 'm_0', 'alpha_m', 'Nn', 'a_ss', 'slope']
        self.params_set = [20., 3225., 0.5, 2.0, 6.0, 0.33, -0.01]
        self.x0 = [self.params_set[2]+0.83, self.params_set[-2]] ## 0.83 correction needed for 100 uM bg
        self.state_bounds = [[0.0, 4.0], [0, 1]]
        self.param_bounds = [[4, 15], [0., 1.], [-0.5, 0.5]] ## N, a_ss, slope
        self.dt = 0.5

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
        df_vec = np.array([dm, (Aa - FR_idx) / self.dt]).T
        return df_vec

    def df_data_generation(self, x, t, p):
        """
        Function to return value of vector field in format used for sp.odeint
        """
        return self.df(t, x, (p, self.eval_stim(t)))

    def eval_stim(self, t):
        return interp1d(self.Tt, self.stim, fill_value='extrapolate')(t)

    def forward_integrate(self):
        """
        Forward integrate the model given true parameters and x0.
        """
        est_params = [self.params_set[i] for i in self.P_idxs]
        self.true_states = odeint(self.df_data_generation, self.x0, self.Tt,
                                  args=(est_params,))

    def sde_df(self, t, x, inputs):
        ## Add noise only to

        p, stim = inputs
        Mm = x[..., 0]
        FR_idx = x[..., 1]
        K_I, K_A, m_0, alpha_m = self.params_set[:4]
        Nn, a_ss, slope = p
        f_c = np.log((1. + stim / K_I) / (1. + stim / K_A))
        f_m = alpha_m * (m_0 - Mm)
        Ee = Nn * (f_m + f_c)
        Aa = 1. / (1. + np.exp(Ee))
        dm = slope * (Aa - a_ss)
        df_vec = np.array([dm, (Aa - FR_idx) / self.dt]).T
        return df_vec

        A = np.array([[-0.5, -2.0],
                      [2.0, -1.0]])

        B = np.diag([0.5, 0.5])
